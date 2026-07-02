"""Depth-based tile-press detection for the web UI (finger/foot on the surface).

A press is detected against a captured per-pixel reference of the EMPTY surface
(background subtraction — works on a tilted surface, no warp needed, because the
depth is D2C-aligned into the same 640x480 space as the colour tiles).

The camera is never perpendicular to the surface and the depth is noisy, so the
detector is built from ADAPTIVE per-pixel maps derived at capture time, plus
temporal gates:

* **Slope-adaptive band** — under an oblique view, a finger touching the surface
  occludes surface that is much farther along the ray, so its apparent height
  can far exceed the finger's thickness. The band's upper edge grows with the
  local surface slope (``max_map``), so tilted regions still detect touches.
* **Noise-adaptive floor** — instead of masking every noisy pixel (which made
  half the frame dead), each pixel's band floor and motion threshold scale with
  its measured noise (``min_map`` / ``motion_map``): noisy zones get stricter,
  clean zones stay sensitive.
* **Artifact mask** — only truly broken pixels (no surface reading, validity
  flicker, extreme noise, or persistently in-band while empty) are masked.
* **Motion gate** — a pixel may only press if its depth changed recently; a
  static scene can never trigger anything.
* **Band stability + confirm + re-arm** — a pixel must stay in-band for
  consecutive frames, a tile fires only after consecutive detections, and after
  release it needs quiet frames before it may fire again.

Pure numpy/cv2 and fully unit-testable with synthetic depth maps.
"""
import cv2
import numpy as np

# --- contact band (mm above the captured surface) ---------------------------
DEFAULT_CONTACT_MIN_MM = 10    # base band floor (raised locally by noise)
DEFAULT_CONTACT_MAX_MM = 35    # base band ceiling (raised locally by slope)
SLOPE_GAIN_PX = 12             # ~finger width in px: ceiling += slope * this
MAX_BAND_CEIL_MM = 150         # never accept more than this above the surface
NOISE_FLOOR_GAIN = 3.0         # band floor = max(base, gain*noise + 5)
NOISE_FLOOR_CAP_MM = 30        # band floor never exceeds this (else zone ~dead)
NOISE_BROKEN_MM = 30           # std above this = hopeless pixel -> artifact

# --- motion gate -------------------------------------------------------------
DEFAULT_MOTION_MM = 10         # base per-pixel change that counts as movement
DEFAULT_MOTION_HOLD = 10       # frames a moved pixel stays "active" (~0.5s)
DEFAULT_BAND_STABLE = 2        # frames a pixel must STAY in-band (kills flicker)

# --- blob shape --------------------------------------------------------------
DEFAULT_MIN_BLOB_PX = 20       # smaller = speckle noise
DEFAULT_MAX_BLOB_FRAC = 0.10   # a finger can't cover >10% of the frame

# --- temporal per-tile logic --------------------------------------------------
DEFAULT_CONFIRM_FRAMES = 2     # consecutive detections before a tile fires
DEFAULT_RELEASE_FRAMES = 3     # consecutive misses before a held tile releases
DEFAULT_REARM_FRAMES = 3       # quiet frames after release before it may re-fire


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


class PressDetector:
    """Stateful press detector: feed it aligned depth frames, get fired tiles.

    Create it from the captured surface, feed the empty capture frames to
    ``calibrate_artifacts`` (builds the noise/slope maps), assign tiles with
    ``set_tiles``, then call ``update(depth)`` once per frame — it returns the
    tile ids newly pressed this frame (edge-triggered; play exactly these).
    """

    def __init__(self, surface_mm,
                 contact_min_mm=DEFAULT_CONTACT_MIN_MM,
                 contact_max_mm=DEFAULT_CONTACT_MAX_MM,
                 motion_mm=DEFAULT_MOTION_MM,
                 motion_hold=DEFAULT_MOTION_HOLD,
                 band_stable=DEFAULT_BAND_STABLE,
                 min_blob_px=DEFAULT_MIN_BLOB_PX,
                 max_blob_frac=DEFAULT_MAX_BLOB_FRAC,
                 confirm_frames=DEFAULT_CONFIRM_FRAMES,
                 release_frames=DEFAULT_RELEASE_FRAMES,
                 rearm_frames=DEFAULT_REARM_FRAMES):
        self.surface = surface_mm.astype(np.int32)
        self.contact_min = int(contact_min_mm)
        self.contact_max = int(contact_max_mm)
        self.motion_mm = int(motion_mm)
        self.motion_hold = int(motion_hold)
        self.band_stable = max(1, int(band_stable))
        self.min_blob_px = int(min_blob_px)
        self.max_blob_frac = float(max_blob_frac)
        self.confirm_frames = max(1, int(confirm_frames))
        self.release_frames = max(1, int(release_frames))
        self.rearm_frames = max(0, int(rearm_frames))

        self.artifact = self.surface <= 0
        self.noise = None            # per-pixel std of the empty scene (mm)
        self._build_maps()
        self.label_map = None
        self.tile_ids = []
        self._reset_runtime()

    # -- adaptive maps ---------------------------------------------------------

    def _build_maps(self):
        """Derive per-pixel band and motion thresholds from surface + noise.

        * ``max_map``: base ceiling + local slope * finger width. On an oblique
          surface the along-ray clearance of a touching finger scales with the
          slope, so a fixed ceiling would swallow real presses.
        * ``min_map`` / ``motion_map``: scale with measured per-pixel noise so
          flicker can't press, without killing whole zones.
        """
        surf = self.surface.astype(np.float32)
        gx = cv2.Sobel(surf, cv2.CV_32F, 1, 0, ksize=3) / 8.0   # mm per pixel
        gy = cv2.Sobel(surf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        slope = cv2.GaussianBlur(cv2.magnitude(gx, gy), (7, 7), 0)
        # Holes fake huge gradients; don't loosen the ceiling near them.
        near_hole = cv2.dilate((self.surface <= 0).astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
        slope[near_hole] = 0.0

        self.max_map = np.clip(self.contact_max + SLOPE_GAIN_PX * slope,
                               self.contact_max, MAX_BAND_CEIL_MM).astype(np.int32)

        if self.noise is not None:
            floor = np.maximum(self.contact_min,
                               NOISE_FLOOR_GAIN * self.noise + 5.0)
            self.min_map = np.clip(floor, self.contact_min,
                                   NOISE_FLOOR_CAP_MM).astype(np.int32)
            self.motion_map = np.maximum(self.motion_mm,
                                         2.5 * self.noise).astype(np.int32)
        else:
            self.min_map = np.full(self.surface.shape, self.contact_min, np.int32)
            self.motion_map = np.full(self.surface.shape, self.motion_mm, np.int32)

    def calibrate_artifacts(self, empty_frames):
        """Learn noise + artifacts from frames of the EMPTY scene.

        With >=3 frames the per-pixel std becomes the noise map (adaptive
        thresholds); only hopeless pixels (validity flicker, extreme noise) are
        masked outright. Pixels still reading in-band while empty — systematic
        mismatch — are masked too. Returns the masked fraction.
        """
        if len(empty_frames) >= 3:
            stack = np.stack([f.astype(np.float32) for f in empty_frames])
            valid_all = (stack > 0).all(axis=0)
            std = np.std(stack, axis=0)
            self.noise = np.where(valid_all, std, 0.0).astype(np.float32)
            self.artifact |= valid_all & (std > NOISE_BROKEN_MM)
            self.artifact |= ~valid_all & (stack > 0).any(axis=0)  # validity flicker
            self._build_maps()  # thresholds now noise-aware

        bad = np.zeros_like(self.artifact)
        for f in empty_frames:
            bad |= self._contact_band(f.astype(np.int32))
        if bad.any():
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.artifact |= bad
        return float(self.artifact.mean())

    def set_tiles(self, label_map, tile_ids):
        self.label_map = label_map
        self.tile_ids = list(tile_ids)
        self._reset_runtime()

    def _reset_runtime(self):
        self._prev = None
        self._motion_age = None
        self._band_age = None
        self._streak = {}
        self._miss = {}
        self._quiet = {}
        self.held = set()

    # -- per-frame ------------------------------------------------------------

    def _contact_band(self, d):
        above = self.surface - d
        return ((d > 0) & (self.surface > 0) & (~self.artifact) &
                (above >= self.min_map) & (above <= self.max_map))

    def update(self, depth_mm):
        """Feed one aligned depth frame; returns the set of tiles newly pressed."""
        if depth_mm is None or self.label_map is None or not self.tile_ids:
            return set()
        d = depth_mm.astype(np.int32)

        # Motion gate: remember which pixels changed recently (vs their own
        # noise level). On a static scene nothing has motion -> nothing presses.
        if self._motion_age is None:
            self._motion_age = np.zeros(d.shape, np.int16)
        else:
            np.subtract(self._motion_age, 1, out=self._motion_age,
                        where=self._motion_age > 0)
        if self._prev is not None:
            moved = (d > 0) & (self._prev > 0) & \
                    (np.abs(d - self._prev) > self.motion_map)
            self._motion_age[moved] = self.motion_hold
        self._prev = d

        # Band stability: a pixel must STAY in the contact band for consecutive
        # frames. Noise flicker jumps in and out; a resting fingertip doesn't.
        band = self._contact_band(d)
        if self._band_age is None:
            self._band_age = np.zeros(d.shape, np.int16)
        self._band_age[~band] = 0
        self._band_age[band] += 1

        contact = band & (self._band_age >= self.band_stable) & (self._motion_age > 0)
        candidates = self._blob_tiles(contact) if contact.any() else set()
        return self._temporal(candidates)

    def _blob_tiles(self, contact):
        """Finger-sized connected blobs -> the one tile each blob covers most."""
        mask = contact.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, blobs, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_area = self.max_blob_frac * mask.shape[0] * mask.shape[1]
        hits = set()
        for b in range(1, n):
            area = stats[b, cv2.CC_STAT_AREA]
            if area < self.min_blob_px or area > max_area:
                continue
            keys = self.label_map[blobs == b]
            keys = keys[keys >= 0]
            if keys.size == 0:
                continue
            counts = np.bincount(keys, minlength=len(self.tile_ids))
            hits.add(self.tile_ids[int(np.argmax(counts))])
        return hits

    def _temporal(self, candidates):
        """Confirm-frames + release-debounce + re-arm quiet period per tile."""
        fired = set()
        for tid in self.tile_ids:
            if tid in candidates:
                self._streak[tid] = self._streak.get(tid, 0) + 1
                self._miss[tid] = 0
                armed = self._quiet.get(tid, self.rearm_frames) >= self.rearm_frames
                if (tid not in self.held and armed
                        and self._streak[tid] >= self.confirm_frames):
                    fired.add(tid)
                    self.held.add(tid)
            else:
                self._streak[tid] = 0
                if tid in self.held:
                    self._miss[tid] = self._miss.get(tid, 0) + 1
                    if self._miss[tid] >= self.release_frames:
                        self.held.discard(tid)
                        self._quiet[tid] = 0
                else:
                    self._quiet[tid] = self._quiet.get(tid, self.rearm_frames) + 1
        return fired


class DepthHitTracker:
    """Edge-trigger + release debounce (kept for the video/brightness path)."""

    def __init__(self, release_frames=3):
        self.release_frames = max(1, int(release_frames))
        self.held = set()
        self._missing = {}

    def update(self, detected):
        detected = set(detected)
        newly = detected - self.held
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
