"""Hardware-free detection & geometry logic for the floor piano.

This module depends on numpy (+ the pure ``constants`` module) only — no camera
SDK, no audio, no GUI — so the core logic can be unit-tested with synthetic
depth frames and developed without a Raspberry Pi or an Orbbec camera attached.

A "depth frame" is always a 2-D numpy array of uint16 millimetre values, where
0 means "no reading" (invalid pixel). After the perspective warp the array is
TARGET_HEIGHT x TARGET_WIDTH and the keys are laid out left-to-right as equal
vertical columns.
"""

from collections import namedtuple

import numpy as np

import constants

# A single key as a rectangle in warped-canvas pixels.
#   kind: "white" | "black"
Key = namedtuple("Key", "name kind x0 y0 x1 y1")


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
    depth = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((height, width)).copy()
    # Some devices (Gemini 335, seen live) mark invalid/overflow pixels as
    # 0xFFFF instead of 0. Fold those into the documented 0 = "no reading"
    # sentinel so floor medians and trigger masks never see bogus 65-m values.
    depth[depth >= 60000] = 0
    return depth


# --- triggering ------------------------------------------------------------

def above_floor_mask(warped_depth, floor_depth, threshold, max_press_height=None):
    """Boolean mask of pixels sitting *above* the floor plane (i.e. a foot).

    A pixel counts only if it has a valid reading (> 0) and is closer to the
    camera than ``floor_depth - threshold`` (the safety buffer in mm).

    With ``max_press_height`` set, pixels HIGHER than that many mm above the
    floor are excluded too — so a foot swinging mid-step, a knee or a torso
    over the mat doesn't press keys, only things near floor level do. Leave it
    None where the full above-floor silhouette is wanted (e.g. masking out a
    standing person before re-sampling the floor depth).
    """
    trigger_depth = floor_depth - threshold
    mask = (warped_depth > 0) & (warped_depth < trigger_depth)
    if max_press_height is not None:
        mask &= warped_depth > floor_depth - max_press_height
    return mask


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


# --- piano keyboard (white + black keys) -----------------------------------

def white_note_name(i, start_octave=None):
    """Name of the i-th white key, e.g. white_note_name(0)='C4', (7)='C5'."""
    start_octave = constants.START_OCTAVE if start_octave is None else start_octave
    return f"{constants.WHITE_SCALE[i % 7]}{start_octave + i // 7}"


def black_note_name(i, start_octave=None):
    """Name of the black key that follows white key i, e.g. (0)='C#4'."""
    start_octave = constants.START_OCTAVE if start_octave is None else start_octave
    return f"{constants.BLACK_AFTER[i % 7]}{start_octave + i // 7}"


def keyboard_note_names(num_white=None, start_octave=None):
    """Ordered key names (all whites, then all blacks) — same order as build_keyboard."""
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    names = [white_note_name(i, start_octave) for i in range(num_white)]
    for i in range(num_white - 1):
        if i % 7 in constants.BLACK_AFTER:
            names.append(black_note_name(i, start_octave))
    return names


def build_keyboard(num_white, width, height, start_octave=None,
                   black_width_ratio=constants.BLACK_WIDTH_RATIO,
                   black_height_ratio=constants.BLACK_HEIGHT_RATIO):
    """Build the list of Key rectangles for a piano keyboard.

    White keys are full-height columns spanning the width. Black keys are narrower,
    cover only the top ``black_height_ratio`` of the mat, and sit centred on the
    boundary between two white keys — but only where a real piano has one (never
    between E-F or B-C). Whites come first in the list, blacks last, so when they
    are painted into a label map the blacks correctly overwrite the whites.
    """
    if num_white <= 0:
        raise ValueError("num_white must be positive")
    bounds = key_bounds(width, num_white)
    keys = [Key(white_note_name(i, start_octave), "white", bounds[i], 0, bounds[i + 1], height)
            for i in range(num_white)]

    white_w = width / num_white
    bw = white_w * black_width_ratio
    bh = round(height * black_height_ratio)
    for i in range(num_white - 1):
        if i % 7 in constants.BLACK_AFTER:
            center = (i + 1) * white_w
            keys.append(Key(black_note_name(i, start_octave), "black",
                            round(center - bw / 2), 0, round(center + bw / 2), bh))
    return keys


def keyboard_label_map(keys, width, height):
    """Paint each key's rectangle into an HxW int map (-1 = no key).

    Keys are painted in list order, so black keys (last) overwrite the white-key
    pixels they overlap — a foot on a black key counts toward the black note only.
    """
    label = np.full((height, width), -1, dtype=np.int32)
    for idx, k in enumerate(keys):
        label[k.y0:k.y1, k.x0:k.x1] = idx
    return label


def detect_hits_labeled(warped_depth, label_map, num_keys, floor_depth, threshold,
                        min_hit_pixels=constants.MIN_HIT_PIXELS):
    """Return the set of key indices whose region has enough above-floor pixels.

    Uses a precomputed label map, so it supports the full 2-D keyboard geometry
    (white + black keys), not just equal columns.
    """
    mask = above_floor_mask(warped_depth, floor_depth, threshold)
    active = set()
    if not mask.any():
        return active
    labels = label_map[mask]
    labels = labels[labels >= 0]
    if labels.size == 0:
        return active
    counts = np.bincount(labels, minlength=num_keys)
    for k in range(num_keys):
        if counts[k] > min_hit_pixels:
            active.add(k)
    return active


def detect_hits_blobs(warped_depth, label_map, num_keys, floor_depth, threshold,
                      min_hit_pixels=constants.MIN_HIT_PIXELS, min_key_overlap=None,
                      max_press_height=constants.MAX_PRESS_HEIGHT, sticky=()):
    """One above-floor blob (= one foot) presses exactly ONE key.

    detect_hits_labeled counts pixels per key over the whole mask, so a single
    foot straddling a boundary fires BOTH keys. Here the mask is split into
    connected components first; each blob big enough to be a foot
    (area > min_hit_pixels) presses only the key holding the largest share of
    it (and only if that share is at least ``min_key_overlap`` pixels, so a
    blob mostly outside the keyboard doesn't fire a key it barely grazes).
    Two feet are two blobs, so chords still work.

    Only pixels within ``max_press_height`` mm of the floor count, so body
    parts passing high above the mat are ignored entirely.

    ``sticky`` is the set of currently-held key indices (the HitTracker's
    state): a blob sitting ~50/50 on a key boundary would otherwise flip its
    argmax with every frame's noise and machine-gun both notes. A held key
    keeps the blob unless a competitor truly dominates it (>~1.4x pixels).

    Needs cv2 (for connectedComponents) — same dependency main.py already has
    for the perspective warp. The mask is morphologically closed first so a
    foot fragmented by depth holes stays ONE blob instead of firing twice.
    """
    import cv2
    if min_key_overlap is None:
        min_key_overlap = max(1, min_hit_pixels // 2)
    mask = above_floor_mask(warped_depth, floor_depth, threshold, max_press_height)
    active = set()
    if not mask.any():
        return active
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    n, blob_labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    for b in range(1, n):  # 0 is background
        if stats[b, cv2.CC_STAT_AREA] <= min_hit_pixels:
            continue
        keys_under = label_map[blob_labels == b]
        keys_under = keys_under[keys_under >= 0]
        if keys_under.size == 0:
            continue
        counts = np.bincount(keys_under, minlength=num_keys)
        k = int(np.argmax(counts))
        if k not in sticky:
            # boundary hysteresis: stay on a held key unless the argmax key
            # truly dominates it. (If the argmax key is itself held, it keeps
            # the blob — never hand it to a weaker held neighbour.)
            held = max((h for h in sticky if h != k and counts[h] >= 0.7 * counts[k]),
                       key=lambda h: counts[h], default=None)
            if held is not None:
                k = held
        if counts[k] >= min_key_overlap:
            active.add(k)
    return active


class HitTracker:
    """Debounce raw per-frame detections into stable held keys.

    A key starts sounding the first frame it is detected (no added latency)
    but is released only after ``release_frames`` consecutive frames without
    it — a noisy mask that drops a key for one frame can no longer retrigger
    the note in a machine-gun loop.
    """

    def __init__(self, release_frames=constants.RELEASE_FRAMES):
        self.release_frames = max(1, int(release_frames))
        self.held = set()
        self._missing = {}

    def update(self, detected):
        """Feed one frame's raw detection set; returns the stable held set."""
        detected = set(detected)
        self.held |= detected
        for k in list(self.held):
            if k in detected:
                self._missing[k] = 0
            else:
                self._missing[k] = self._missing.get(k, 0) + 1
                if self._missing[k] >= self.release_frames:
                    self.held.discard(k)
                    del self._missing[k]
        return set(self.held)

    def reset(self):
        self.held = set()
        self._missing = {}


def suppress_white_under_black(active, keys):
    """Drop white keys that overlap an active black key.

    A foot on a black key inevitably covers some neighbouring white-key area; this
    keeps the press to the intended (black) note instead of sounding a cluster.
    Returns a new set.
    """
    out = set(active)
    for b in active:
        if keys[b].kind != "black":
            continue
        bk = keys[b]
        for i in list(out):
            k = keys[i]
            if k.kind == "white" and k.x0 < bk.x1 and k.x1 > bk.x0:
                out.discard(i)
    return out


def note_filename(note):
    """Map a note name to its sample filename ('#'' -> 's'): 'C#4' -> 'Cs4.wav'."""
    return note.replace("#", "s") + ".wav"


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


def sample_floor_depth(depth_array, center_xy, rgb_shape, patch=10):
    """Median floor depth in a small patch around the mat centre, or None.

    ``center_xy`` is in RGB-image coordinates; it is scaled into depth space.
    Returns None when the point is too close to the edge or no valid depth
    pixels exist there — callers must treat that as "not calibratable right
    now", NOT silently substitute a default (a wrong floor depth makes the
    piano dead or constantly firing).
    """
    dx, dy = scale_point_to_depth(center_xy, rgb_shape, depth_array.shape)
    dx, dy = int(dx), int(dy)
    dh, dw = depth_array.shape
    if patch <= dy < dh - patch and patch <= dx < dw - patch:
        sample = depth_array[dy - patch:dy + patch, dx - patch:dx + patch]
        valid = sample[sample > 0]
        if valid.size > 0:
            return int(np.median(valid))
    return None


def build_config(corner_points, floor_depth, rgb_shape,
                 num_white=None, start_octave=None,
                 trigger_threshold=constants.DEFAULT_TRIGGER_THRESHOLD):
    """Assemble the config.json dict from detected corners.

    ``corner_points`` maps each CORNER_ID -> (x, y) array. Corners are ordered
    0,1,2,3 (TL,TR,BR,BL) to match the perspective-warp destination in main.py.
    The keyboard is described by ``num_white`` (white-key count); names + black
    keys are derived from it at runtime via build_keyboard().
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    start_octave = constants.START_OCTAVE if start_octave is None else start_octave
    rh, rw = rgb_shape[:2]
    corners = [list(map(float, corner_points[mid])) for mid in constants.CORNER_IDS]
    return {
        "corners": corners,
        "num_white_keys": int(num_white),
        "start_octave": int(start_octave),
        "floor_depth": int(floor_depth),
        "trigger_threshold": int(trigger_threshold),
        "canvas_size": [rw, rh],
    }


# --- config validation -----------------------------------------------------

REQUIRED_CONFIG_KEYS = ("corners", "num_white_keys", "floor_depth", "trigger_threshold")


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
        for v in pt:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"config corner {i} has a non-numeric coordinate: {pt}")
    # Shoelace area of the TL,TR,BR,BL polygon. Zero/near-zero = degenerate
    # (repeated or collinear corners); negative = wrong winding vs the warp
    # destination (markers swapped / mirrored placement) — both would produce
    # a folded or mirrored key grid with no visible error later.
    pts = cfg["corners"]
    area2 = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]
                for i in range(4))
    if area2 < 200:  # signed double-area, image coords (y down): TL,TR,BR,BL is positive
        raise ValueError(
            "config 'corners' do not form a valid TL,TR,BR,BL quad "
            f"(signed area {area2 / 2:.0f}) — markers swapped or calibration degenerate?"
        )
    if not isinstance(cfg["num_white_keys"], int) or isinstance(cfg["num_white_keys"], bool) \
            or cfg["num_white_keys"] <= 0:
        raise ValueError(f"config 'num_white_keys' must be a positive integer, got {cfg['num_white_keys']}")
    if "start_octave" in cfg and (not isinstance(cfg["start_octave"], int)
                                  or isinstance(cfg["start_octave"], bool)):
        raise ValueError(f"config 'start_octave' must be an integer, got {cfg['start_octave']}")
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
    """Build synthetic frames where a 'foot' visits each equal-column key in turn.

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


def stamp_key(frame, key, depth):
    """Return a copy of ``frame`` with ``depth`` written over a Key's rectangle."""
    out = frame.copy()
    out[key.y0:key.y1, key.x0:key.x1] = depth
    return out


def keyboard_sweep_frames(keys, width, height, floor_depth=constants.DEFAULT_FLOOR_DEPTH,
                          foot_depth=None, hold=3, gap=2):
    """Build frames where a 'foot' visits each Key (white then black) in turn.

    Each frame triggers exactly one key. For a black key the foot fills the black
    rectangle. For a white key the foot is placed only in the *lower* band that no
    black key overlaps, so the white note triggers cleanly instead of being
    swallowed by suppress_white_under_black(). Drives the 24-key demo/tests.
    """
    if foot_depth is None:
        foot_depth = floor_depth - 200
    base = flat_floor_frame(height, width, floor_depth)
    black_h = max((k.y1 for k in keys if k.kind == "black"), default=0)
    frames = []
    for key in keys:
        frame = base.copy()
        if key.kind == "white":
            frame[black_h:key.y1, key.x0:key.x1] = foot_depth   # pure white area
        else:
            frame[key.y0:key.y1, key.x0:key.x1] = foot_depth
        frames.extend([frame] * hold)
        frames.extend([base] * gap)
    return frames
