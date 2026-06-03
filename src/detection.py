"""Hardware-free detection & geometry logic for the floor piano.

This module depends on numpy (+ the pure ``constants`` module) only — no camera
SDK, no audio, no GUI — so the core logic can be unit-tested with synthetic
depth frames and developed without a Raspberry Pi or an Orbbec camera attached.

A "depth frame" is always a 2-D numpy array of uint16 millimetre values, where
0 means "no reading" (invalid pixel). After the perspective warp the array is
TARGET_HEIGHT x TARGET_WIDTH and the keys are laid out left-to-right as equal
vertical columns.
"""

import numpy as np

import constants


# --- key geometry ----------------------------------------------------------

def key_width(target_width, num_keys):
    """Nominal pixel width of a single key column (floor division)."""
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    return target_width // num_keys


def key_bounds(target_width, num_keys):
    """Column boundaries [x0, x1, ..., xN] for the key zones.

    Uses rounding so the columns tile the *entire* width with no silent gap at
    the right edge when target_width is not divisible by num_keys.
    """
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    return [round(i * target_width / num_keys) for i in range(num_keys + 1)]


# --- frame decoding --------------------------------------------------------

def decode_depth(raw_bytes, height, width):
    """Decode a raw 16-bit depth buffer into an HxW uint16 array.

    Returns None if the buffer size does not match a 16-bit height*width frame
    (e.g. the camera delivered an unexpected pixel format), so callers can skip
    the frame instead of crashing on a reshape.
    """
    # Check the raw byte count first: np.frombuffer itself raises if the buffer
    # is not a whole number of uint16 elements, so we cannot rely on .size.
    if memoryview(raw_bytes).nbytes != height * width * 2:
        return None
    # .copy() so we own a contiguous, writable array: the camera SDK may recycle
    # its frame buffer on the next read, and numpy 2.x is strict about contiguity.
    return np.frombuffer(raw_bytes, dtype=np.uint16).reshape((height, width)).copy()


# --- triggering ------------------------------------------------------------

def above_floor_mask(warped_depth, floor_depth, threshold):
    """Boolean mask of pixels sitting *above* the floor plane (i.e. a foot).

    A pixel counts only if it has a valid reading (> 0) and is closer to the
    camera than ``floor_depth - threshold`` (the safety buffer in mm).
    """
    trigger_depth = floor_depth - threshold
    return (warped_depth > 0) & (warped_depth < trigger_depth)


def detect_hits(warped_depth, num_keys, floor_depth, threshold,
                min_hit_pixels=constants.MIN_HIT_PIXELS):
    """Return the set of key *indices* (0-based) that are currently pressed.

    A key fires when its column contains more than ``min_hit_pixels`` above-floor
    pixels. Pure: same input -> same output, no side effects.
    """
    mask = above_floor_mask(warped_depth, floor_depth, threshold)
    active = set()
    if not mask.any():
        return active
    bounds = key_bounds(warped_depth.shape[1], num_keys)
    for i in range(num_keys):
        zone = mask[:, bounds[i]:bounds[i + 1]]
        if int(zone.sum()) > min_hit_pixels:
            active.add(i)
    return active


def newly_pressed(current, previous):
    """Edge-trigger helper: notes present now but not in the previous frame."""
    return set(current) - set(previous)


def median_floor_depth(warped_depth, min_valid_pixels=1000):
    """Median of the valid depth readings, or None if there is too little data."""
    valid = warped_depth[warped_depth > 0]
    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


# --- calibration helpers (pure, so they are unit-testable) -----------------

def scale_point_to_depth(point_xy, rgb_shape, depth_shape):
    """Map an (x, y) pixel from RGB-image space into depth-image space by resolution.

    NOTE: this only accounts for resolution, not the different FOV/origin of the
    RGB vs depth sensor — see CODE_REVIEW finding #2. It is correct for sampling a
    rough central depth, but not for a precise per-key registration.
    """
    rh, rw = rgb_shape[:2]
    dh, dw = depth_shape[:2]
    return (point_xy[0] * dw / rw, point_xy[1] * dh / rh)


def sample_floor_depth(depth_array, center_xy, rgb_shape,
                       default=constants.DEFAULT_FLOOR_DEPTH, patch=10):
    """Median floor depth in a small patch around the mat centre.

    ``center_xy`` is in RGB-image coordinates; it is scaled into depth space.
    Returns ``default`` if the point is too close to the edge or no valid depth
    pixels are found.
    """
    dx, dy = scale_point_to_depth(center_xy, rgb_shape, depth_array.shape)
    dx, dy = int(dx), int(dy)
    dh, dw = depth_array.shape
    if patch <= dy < dh - patch and patch <= dx < dw - patch:
        sample = depth_array[dy - patch:dy + patch, dx - patch:dx + patch]
        valid = sample[sample > 0]
        if valid.size > 0:
            return int(np.median(valid))
    return default


def build_config(corner_points, floor_depth, rgb_shape,
                 keys=None, trigger_threshold=constants.DEFAULT_TRIGGER_THRESHOLD):
    """Assemble the config.json dict from detected corners.

    ``corner_points`` maps each CORNER_ID -> (x, y) array. Corners are ordered
    0,1,2,3 (TL,TR,BR,BL) to match the perspective-warp destination in main.py.
    """
    rh, rw = rgb_shape[:2]
    corners = [list(map(float, corner_points[mid])) for mid in constants.CORNER_IDS]
    return {
        "corners": corners,
        "keys": list(keys) if keys else list(constants.DEFAULT_KEYS),
        "floor_depth": int(floor_depth),
        "trigger_threshold": int(trigger_threshold),
        "canvas_size": [rw, rh],
    }


# --- config validation -----------------------------------------------------

REQUIRED_CONFIG_KEYS = ("corners", "keys", "floor_depth", "trigger_threshold")


def validate_config(cfg):
    """Validate a parsed config dict, raising ValueError on the first problem."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            raise ValueError(f"config is missing required key: '{key}'")
    if len(cfg["corners"]) != 4:
        raise ValueError(f"config 'corners' must have exactly 4 points, got {len(cfg['corners'])}")
    for i, pt in enumerate(cfg["corners"]):
        if len(pt) != 2:
            raise ValueError(f"config corner {i} must be [x, y], got {pt}")
    if not cfg["keys"]:
        raise ValueError("config 'keys' must be a non-empty list")
    for k in ("floor_depth", "trigger_threshold"):
        if not isinstance(cfg[k], (int, float)) or isinstance(cfg[k], bool):
            raise ValueError(f"config '{k}' must be a number, got {type(cfg[k]).__name__}")
    if cfg["floor_depth"] <= cfg["trigger_threshold"]:
        raise ValueError(
            "config 'floor_depth' must be greater than 'trigger_threshold' "
            f"({cfg['floor_depth']} <= {cfg['trigger_threshold']})"
        )
    return True


# --- test / dev helpers ----------------------------------------------------

class MockDepthSource:
    """A stand-in for a real depth camera that replays preset frames.

    Lets you develop and test the main loop with no hardware attached. Each frame
    is a 2-D uint16 numpy array, exactly like a real warped/raw frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        for f in self._frames:
            yield f

    def __len__(self):
        return len(self._frames)


def flat_floor_frame(height, width, floor_depth=constants.DEFAULT_FLOOR_DEPTH, dtype=np.uint16):
    """A synthetic frame of pure floor at a constant depth (a 'no feet' scene)."""
    return np.full((height, width), floor_depth, dtype=dtype)


def stamp_foot(frame, key_index, num_keys, foot_depth, rows=None):
    """Return a copy of ``frame`` with a 'foot' (shallower depth) over one key."""
    out = frame.copy()
    bounds = key_bounds(frame.shape[1], num_keys)
    r0, r1 = (0, frame.shape[0]) if rows is None else rows
    out[r0:r1, bounds[key_index]:bounds[key_index + 1]] = foot_depth
    return out


def sweep_frames(height, width, num_keys, floor_depth=constants.DEFAULT_FLOOR_DEPTH,
                 foot_depth=None, hold=3, gap=2):
    """Build synthetic frames where a 'foot' visits each key in turn.

    For each key: ``hold`` frames with the foot present, then ``gap`` empty floor
    frames (so the edge-trigger re-arms). Drives the camera-free demo/tests.
    """
    if foot_depth is None:
        foot_depth = floor_depth - 200
    base = flat_floor_frame(height, width, floor_depth)
    frames = []
    for k in range(num_keys):
        foot = stamp_foot(base, k, num_keys, foot_depth)
        frames.extend([foot] * hold)
        frames.extend([base] * gap)
    return frames
