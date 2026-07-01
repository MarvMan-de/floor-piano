"""Depth-based tile-press detection for the web UI (finger/foot on the surface).

Unlike the standalone piano (flat floor, single floor_depth), the test rig is a
tilted tablet, so a single depth threshold won't do. Instead we use per-pixel
**background subtraction**: capture the empty surface once, then a "press" is any
valid pixel that is closer to the camera than (surface - threshold) — a finger
sticking up off the surface. This works at any tilt and needs no warp, because
the depth is D2C-aligned into the same 640x480 space as the colour tiles.

Pure numpy/cv2 so it is unit-testable with synthetic depth maps.
"""
import cv2
import numpy as np

# Millimetre thresholds — fingers are shallow, so keep threshold just above the
# depth noise floor. Tunable.
DEFAULT_THRESHOLD_MM = 20      # min height above the surface to count as a press
DEFAULT_MAX_HEIGHT_MM = 200    # ignore things far above (hand/arm passing over)
DEFAULT_MIN_PIXELS = 40        # min press pixels inside a tile to fire it


def build_tile_label_map(tiles, width, height):
    """Paint each enabled tile polygon into an int32 label map (-1 = no tile).

    Painted in list order, so later tiles (black keys) overwrite the whites they
    overlap — same convention as detection.keyboard_label_map. Returns
    (label_map, tile_ids) where label_map holds indices into tile_ids.
    """
    label = np.full((height, width), -1, dtype=np.int32)
    tile_ids = []
    for tile in tiles:
        if not tile.get("enabled", True):
            continue
        poly = tile.get("polygon") or []
        if len(poly) < 3:
            continue
        idx = len(tile_ids)
        cv2.fillPoly(label, [np.asarray(poly, dtype=np.int32)], int(idx))
        tile_ids.append(tile["id"])
    return label, tile_ids


def detect_tile_hits_depth(depth_mm, surface_mm, label_map, tile_ids,
                           threshold_mm=DEFAULT_THRESHOLD_MM,
                           max_height_mm=DEFAULT_MAX_HEIGHT_MM,
                           min_pixels=DEFAULT_MIN_PIXELS):
    """Return the set of tile ids currently pressed.

    A pixel is "pressed" when it and the surface both have a valid reading and it
    sits between ``threshold_mm`` and ``max_height_mm`` in front of the surface.
    Per tile, more than ``min_pixels`` such pixels = pressed.
    """
    if surface_mm is None or depth_mm is None:
        return set()
    d = depth_mm.astype(np.int32)
    s = surface_mm.astype(np.int32)
    press = (d > 0) & (s > 0) & (d < s - threshold_mm) & (d > s - max_height_mm)
    if not press.any():
        return set()
    labels = label_map[press]
    labels = labels[labels >= 0]
    if labels.size == 0:
        return set()
    counts = np.bincount(labels, minlength=len(tile_ids))
    return {tile_ids[i] for i in range(len(tile_ids)) if counts[i] > min_pixels}


class DepthHitTracker:
    """Edge-trigger + release debounce so a held press fires its note only once.

    ``update(detected)`` returns the set of tile ids that transitioned to pressed
    this frame (the notes to play now). A tile stays "held" until it has been
    absent for ``release_frames`` consecutive frames, so depth noise dropping it
    for a frame doesn't machine-gun the note.
    """

    def __init__(self, release_frames=3):
        self.release_frames = max(1, int(release_frames))
        self.held = set()
        self._missing = {}

    def update(self, detected):
        detected = set(detected)
        newly = detected - self.held        # rising edge -> fire these
        self.held |= detected
        for k in list(self.held):
            if k in detected:
                self._missing[k] = 0
            else:
                self._missing[k] = self._missing.get(k, 0) + 1
                if self._missing[k] >= self.release_frames:
                    self.held.discard(k)
                    self._missing.pop(k, None)
        return newly

    def reset(self):
        self.held = set()
        self._missing = {}
