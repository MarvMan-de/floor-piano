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

# Millimetre "contact band" above the captured surface. A press = something right
# AT the surface (a fingertip, whose top sits ~finger-thickness above it). A hand
# hovering / passing over is much higher and falls OUTSIDE the band, so it does
# NOT trigger — this is the fix for "the note plays when my hand just passes over".
DEFAULT_CONTACT_MIN_MM = 5     # ignore depth noise right at the surface plane
DEFAULT_CONTACT_MAX_MM = 30    # fingertip touching; above this = hovering -> ignore
DEFAULT_MIN_PIXELS = 20        # min contact pixels inside a tile to fire it


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
                           contact_min_mm=DEFAULT_CONTACT_MIN_MM,
                           contact_max_mm=DEFAULT_CONTACT_MAX_MM,
                           min_pixels=DEFAULT_MIN_PIXELS):
    """Return the set of tile ids currently *touched*.

    ``above = surface - depth`` is how far a pixel sits in front of the captured
    surface. A **touch** is a pixel in the thin contact band
    ``contact_min_mm <= above <= contact_max_mm`` — i.e. right at the surface (a
    fingertip). A hovering / passing hand is much further in front (above >
    contact_max_mm) and is ignored, so it no longer triggers the note. Per tile,
    more than ``min_pixels`` contact pixels = pressed.
    """
    if surface_mm is None or depth_mm is None:
        return set()
    d = depth_mm.astype(np.int32)
    s = surface_mm.astype(np.int32)
    above = s - d
    touch = (d > 0) & (s > 0) & (above >= contact_min_mm) & (above <= contact_max_mm)
    if not touch.any():
        return set()
    labels = label_map[touch]
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
