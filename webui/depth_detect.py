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
* **Attachment gate** — a real press is part of a finger/foot/leg that sticks
  up OUT of the contact band right next to the contact patch. Depth wobble
  from moving shadows (sunlight IR interference) lies flat on the surface with
  nothing above it — such orphan blobs are rejected. Blobs touching the frame
  border are exempt (the leg may be outside the view).
* **Ring guard** — the sloped SIDES of a foot that merely stands next to a key
  always cut through the contact band as a thin ring hugging the hovering
  foot. Contact pixels within a few px of hover pixels are discarded: a real
  press reaches well beyond its own limb's outline, a grazing edge doesn't.
* **Centroid trigger** — each surviving blob fires the tile under its
  CENTRE (the middle of the foot), not whichever tile its rim grazes.
* **Band stability + confirm + re-arm** — a pixel must stay in-band for
  consecutive frames, a tile fires only after consecutive detections, and after
  release it needs quiet frames before it may fire again.

* **Plane fallback** — where the measured reference is unusable (holes, flicker,
  specular reflections on the mat) a plane fitted to the good mat pixels stands
  in for it instead of masking the zone dead (``plane_fallback``, default on).
  The plane is fitted in INVERSE depth (1/z is linear in image coordinates for
  any camera tilt). Limitation: where the LIVE depth is 0 while the foot is
  down, the pixel stays blind — the plane only repairs the reference.

Pure numpy/cv2 and fully unit-testable with synthetic depth maps.
"""
from typing import NamedTuple, Optional

import cv2
import numpy as np

# --- contact band (mm above the captured surface) ---------------------------
DEFAULT_CONTACT_MIN_MM = 10    # base band floor (raised locally by noise)
DEFAULT_CONTACT_MAX_MM = 35    # base band ceiling (raised locally by slope)
SLOPE_GAIN_PX = 12             # ~finger width in px: ceiling += slope * this
MAX_BAND_CEIL_MM = 80          # never accept more than this above the surface —
                               # was 150, which let a foot hovering ~15-20cm fire
                               # on sloped zones; with a roughly top-down camera
                               # 80mm still covers real presses comfortably
NOISE_FLOOR_GAIN = 3.0         # band floor = max(base, gain*noise + 5)
NOISE_FLOOR_CAP_MM = 30        # band floor never exceeds this (else zone ~dead)
NOISE_BROKEN_MM = 30           # std above this = hopeless pixel -> artifact

# --- plane fallback (rescue of unusable reference pixels) -------------------
PLANE_TOL_MM = 15              # measured ref within this of the plane = usable
PLANE_CONTACT_MIN_MM = 20      # rescued px: stricter band floor (plane is rougher)
PLANE_CONSIST_MM = 10          # empty frames read this far IN FRONT of the plane
                               # = in-band garbage, never rescue it
PLANE_RANSAC_TOL_MM = 10
PLANE_RANSAC_ITERS = 150
PLANE_SAMPLE_PX = 5000
PLANE_MIN_FIT_PX = 2000
PLANE_MIN_INLIER_RATIO = 0.6
PLANE_MAX_RMS_MM = 6.0
PLANE_FIT_NOISE_MM = 10        # only low-noise pixels take part in the fit
PLANE_Z_RANGE_MM = (200, 10000)
MAT_DILATE_PX = 15             # tiles dilated by this = the "mat" domain

# --- motion gate -------------------------------------------------------------
DEFAULT_MOTION_MM = 10         # base per-pixel change that counts as movement
DEFAULT_MOTION_HOLD = 10       # frames a moved pixel stays "active" (~0.5s)
DEFAULT_BAND_STABLE = 2        # frames a pixel must STAY in-band (kills flicker)

# --- blob shape --------------------------------------------------------------
DEFAULT_MIN_BLOB_PX = 20       # smaller = speckle noise
DEFAULT_MAX_BLOB_FRAC = 0.10   # a finger can't cover >10% of the frame
ATTACH_RADIUS_PX = 15          # how far around a blob to look for the limb above
ATTACH_MIN_HOVER_PX = 8        # this many hover px near the blob = attached
RING_GUARD_PX = 8              # contact this close to hover = foot edge / flying
                               # pixels (stereo edge halo), not a press
PRESENT_MIN_PX = 25            # in-band px over a tile that count as "foot still
                               # there" — below this it's speckle noise, which
                               # must not keep a released key held forever
HELD_OVERLAP_FRAC = 0.25       # blob overlapping a held tile this much = the
                               # same foot rolling/spreading, not a new press
FOOT_OBJ_MAX_MM = 250          # pixels up to this above the surface form the
                               # "foot object"; the cut keeps two feet separate
                               # (they would otherwise connect through the body)
FOOT_OBJ_MIN_MM = 12           # object floor — below this it's mat wrinkles /
                               # noise, which must not weld feet together
PRESENCE_RADIUS_PX = 45        # band pixels only count as "foot still there"
                               # within this range of hovering pixels — a mat
                               # wrinkle left behind by a press has no foot
                               # above it and must not keep a key held forever

# --- temporal per-tile logic --------------------------------------------------
DEFAULT_CONFIRM_FRAMES = 2     # consecutive detections before a tile fires
DEFAULT_RELEASE_FRAMES = 3     # consecutive misses before a held tile releases
DEFAULT_REARM_FRAMES = 3       # quiet frames after release before it may re-fire


class _Maps(NamedTuple):
    """Immutable bundle of the per-pixel maps — swapped in ONE assignment so the
    capture thread never sees a half-rebuilt set."""
    surface: np.ndarray
    artifact: np.ndarray
    min_map: np.ndarray
    max_map: np.ndarray
    motion_map: np.ndarray
    rescued: Optional[np.ndarray]
    hole_rescued: Optional[np.ndarray]


def fit_inverse_depth_plane(depth, mask, rng=None):
    """RANSAC plane through ``depth`` (mm) at the True pixels of ``mask``.

    For a pinhole camera a 3D plane gives 1/z = a*u + b*v + c exactly (any
    tilt), so the fit is done on w = 1000/z with u, v normalised to 0..1.
    Inliers are judged in mm. Returns {"coef", "inlier_ratio", "rms_mm",
    "n_fit"} or None when there is no convincing plane.
    """
    h, w_img = depth.shape
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs].astype(np.float64)
    ok = (z >= PLANE_Z_RANGE_MM[0]) & (z <= PLANE_Z_RANGE_MM[1])
    ys, xs, z = ys[ok], xs[ok], z[ok]
    n = z.size
    if n < PLANE_MIN_FIT_PX:
        return None
    rng = rng if rng is not None else np.random.default_rng(0)
    A = np.stack([xs / w_img, ys / h, np.ones(n)], axis=1)
    wv = 1000.0 / z
    sub = rng.choice(n, min(n, PLANE_SAMPLE_PX), replace=False)
    As, zs = A[sub], z[sub]
    idx = rng.integers(0, sub.size, (PLANE_RANSAC_ITERS, 3))
    M = As[idx]
    good = np.abs(np.linalg.det(M)) > 1e-9
    if not good.any():
        return None
    coefs = np.linalg.solve(M[good], wv[sub][idx][good][..., None])[..., 0]
    pw = coefs @ As.T
    with np.errstate(divide="ignore", invalid="ignore"):
        zhat = np.where(pw > 0, 1000.0 / pw, np.inf)
    counts = (np.abs(zs[None, :] - zhat) < PLANE_RANSAC_TOL_MM).sum(axis=1)
    coef = coefs[int(np.argmax(counts))]

    def _inliers(c):
        pw_ = A @ c
        with np.errstate(divide="ignore", invalid="ignore"):
            zh = np.where(pw_ > 0, 1000.0 / pw_, np.inf)
        return np.abs(z - zh) < PLANE_RANSAC_TOL_MM

    inl = _inliers(coef)
    if inl.sum() < 3:
        return None
    sq = z[inl]                      # weights z^2: dz ~ z^2 * dw / 1000
    coef, *_ = np.linalg.lstsq(A[inl] * sq[:, None], wv[inl] * sq, rcond=None)
    inl = _inliers(coef)
    if inl.sum() < 3:
        return None
    pw = A[inl] @ coef
    if (pw <= 0).any():
        return None
    rms = float(np.sqrt(np.mean((z[inl] - 1000.0 / pw) ** 2)))
    ratio = float(inl.sum()) / n
    if ratio < PLANE_MIN_INLIER_RATIO or rms > PLANE_MAX_RMS_MM:
        return None
    return {"coef": tuple(float(c) for c in coef), "inlier_ratio": ratio,
            "rms_mm": rms, "n_fit": int(inl.sum())}


def plane_depth_map(coef, shape):
    """Depth (int32 mm) of the fitted plane at every pixel; 0 where invalid."""
    h, w = shape
    a, b, c = coef
    wv = (a * (np.arange(w) / w)[None, :] + b * (np.arange(h) / h)[:, None] + c)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(wv > 1e-6, 1000.0 / wv, 0.0)
    z[(z < PLANE_Z_RANGE_MM[0]) | (z > PLANE_Z_RANGE_MM[1])] = 0.0
    return np.round(z).astype(np.int32)


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
                 rearm_frames=DEFAULT_REARM_FRAMES,
                 require_attached=True,
                 plane_fallback=True):
        self._measured = surface_mm.astype(np.int32)
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
        self.require_attached = bool(require_attached)
        self.plane_fallback = bool(plane_fallback)

        self._raw_artifact = self._measured <= 0
        self.noise = None            # per-pixel std of the empty scene (mm)
        self._rescued = None         # plane stands in for the reference here
        self._hole_rescued = None    # ... and the empty scene had 0-readings
        self._plane_z = None
        self._mat_mask = None
        self._empty_closest = None   # per-pixel nearest valid reading (empty)
        self._valid_all = None
        self._calibrated = False
        self.plane_info = {"enabled": self.plane_fallback, "ok": False}
        self._build_maps()
        self.label_map = None
        self.tile_ids = []
        self._reset_runtime()

    # -- adaptive maps ---------------------------------------------------------

    surface = property(lambda self: self._maps.surface)
    artifact = property(lambda self: self._maps.artifact)
    min_map = property(lambda self: self._maps.min_map)
    max_map = property(lambda self: self._maps.max_map)
    motion_map = property(lambda self: self._maps.motion_map)
    rescued = property(lambda self: self._maps.rescued)

    def _build_maps(self):
        """Derive per-pixel band and motion thresholds from surface + noise.

        * ``max_map``: base ceiling + local slope * finger width. On an oblique
          surface the along-ray clearance of a touching finger scales with the
          slope, so a fixed ceiling would swallow real presses.
        * ``min_map`` / ``motion_map``: scale with measured per-pixel noise so
          flicker can't press, without killing whole zones.
        """
        rescued = self._rescued
        if rescued is not None:
            surface = np.where(rescued, self._plane_z, self._measured)
            artifact = self._raw_artifact & ~rescued
        else:
            surface = self._measured
            artifact = self._raw_artifact
        surf = surface.astype(np.float32)
        gx = cv2.Sobel(surf, cv2.CV_32F, 1, 0, ksize=3) / 8.0   # mm per pixel
        gy = cv2.Sobel(surf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        slope = cv2.GaussianBlur(cv2.magnitude(gx, gy), (7, 7), 0)
        # Holes fake huge gradients; don't loosen the ceiling near them.
        near_hole = cv2.dilate((self._measured <= 0).astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
        slope[near_hole] = 0.0
        if rescued is not None:
            # at the seam measured<->plane the Sobel sees a fake step; use the
            # plane's own (smooth) slope there instead
            pz = self._plane_z.astype(np.float32)
            pgx = cv2.Sobel(pz, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            pgy = cv2.Sobel(pz, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            seam = cv2.dilate(rescued.astype(np.uint8),
                              np.ones((5, 5), np.uint8)) > 0
            slope[seam] = cv2.magnitude(pgx, pgy)[seam]

        # calibrate_touch() may raise contact_max above the static cap — the
        # cap then follows, so a calibrated ceiling is never clipped away
        ceil_cap = max(MAX_BAND_CEIL_MM, self.contact_max + 20)
        max_map = np.clip(self.contact_max + SLOPE_GAIN_PX * slope,
                          self.contact_max, ceil_cap).astype(np.int32)

        if self.noise is not None:
            floor = np.maximum(self.contact_min,
                               NOISE_FLOOR_GAIN * self.noise + 5.0)
            min_map = np.clip(floor, self.contact_min,
                              NOISE_FLOOR_CAP_MM).astype(np.int32)
            motion_map = np.maximum(self.motion_mm,
                                    2.5 * self.noise).astype(np.int32)
        else:
            min_map = np.full(surface.shape, self.contact_min, np.int32)
            motion_map = np.full(surface.shape, self.motion_mm, np.int32)
        hole_rescued = self._hole_rescued
        if rescued is not None:
            # the plane is rougher than a real measurement: stricter floor, and
            # the noise measured on garbage readings means nothing there
            min_map = np.where(rescued, np.maximum(min_map, PLANE_CONTACT_MIN_MM),
                               min_map).astype(np.int32)
            max_map = np.where(rescued, np.maximum(max_map, min_map + 15),
                               max_map).astype(np.int32)
            motion_map = np.where(rescued, self.motion_mm,
                                  motion_map).astype(np.int32)
        self._maps = _Maps(surface.astype(np.int32), artifact, min_map, max_map,
                           motion_map, rescued, hole_rescued)

    def calibrate_artifacts(self, empty_frames):
        """Learn noise + artifacts from frames of the EMPTY scene.

        With >=3 frames the per-pixel std becomes the noise map (adaptive
        thresholds); only hopeless pixels (validity flicker, extreme noise) are
        masked outright. Pixels still reading in-band while empty — systematic
        mismatch — are masked too. Returns the masked fraction.
        """
        if self.plane_fallback:      # recalibration starts from the raw state
            self._rescued = self._hole_rescued = self._plane_z = None
        self._valid_all = None
        if len(empty_frames) >= 3:
            stack = np.stack([f.astype(np.float32) for f in empty_frames])
            valid_all = (stack > 0).all(axis=0)
            self._valid_all = valid_all
            std = np.std(stack, axis=0)
            self.noise = np.where(valid_all, std, 0.0).astype(np.float32)
            self._raw_artifact = self._raw_artifact | (valid_all & (std > NOISE_BROKEN_MM))
            self._raw_artifact = self._raw_artifact | (~valid_all & (stack > 0).any(axis=0))  # validity flicker
            self._build_maps()  # thresholds now noise-aware

        bad = np.zeros_like(self._raw_artifact)
        for f in empty_frames:
            bad |= self._contact_band(f.astype(np.int32))
        if bad.any():
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self._raw_artifact = self._raw_artifact | bad

        # nearest valid reading per pixel over the empty frames: a pixel that
        # ever read well IN FRONT of the plane is in-band garbage, not a hole
        if empty_frames:
            st = np.stack([f.astype(np.int32) for f in empty_frames])
            closest = np.where(st > 0, st, 1 << 30).min(axis=0)
            self._empty_closest = np.where(closest == 1 << 30, 0, closest)
        self._calibrated = True
        if self.plane_fallback:
            self._update_plane()
        else:
            self._build_maps()
        return float(self.artifact.mean())

    # -- plane fallback --------------------------------------------------------

    def _mat_mask_from_labels(self):
        if self.label_map is None or not (self.label_map >= 0).any():
            return None
        k = 2 * MAT_DILATE_PX + 1
        return cv2.dilate((self.label_map >= 0).astype(np.uint8),
                          np.ones((k, k), np.uint8)) > 0

    def _update_plane(self):
        """Fit the plane and decide which unusable pixels it may stand in for.

        A pixel the plane cannot help (or any failure) keeps its old fate, so
        this can only ever improve on the plain per-pixel reference."""
        self._mat_mask = mat = self._mat_mask_from_labels()
        meas, raw = self._measured, self._raw_artifact
        cand = (meas > 0) & ~raw
        if mat is not None:
            cand &= mat
        if self.noise is not None:
            cand &= self.noise <= PLANE_FIT_NOISE_MM
        fit = fit_inverse_depth_plane(meas, cand)
        if fit is None:
            self._rescued = self._hole_rescued = self._plane_z = None
            self.plane_info = {"enabled": True, "ok": False,
                               "reason": "no convincing plane"}
            self._build_maps()
            return
        plane_z = plane_depth_map(fit["coef"], meas.shape)
        usable = (meas > 0) & ~raw & (np.abs(meas - plane_z) <= PLANE_TOL_MM)
        if mat is not None:
            domain = mat
        else:   # no tiles yet: the hull of the good pixels (keeps letterbox out)
            pts = cv2.findNonZero(cand.astype(np.uint8))
            domain = np.zeros(meas.shape, np.uint8)
            cv2.fillConvexPoly(domain, cv2.convexHull(pts), 1)
            domain = domain > 0
        rescued = domain & ~usable & (plane_z > 0)
        if self._empty_closest is not None:
            c = self._empty_closest
            rescued &= ~((c > 0) & (plane_z - c >= PLANE_CONSIST_MM))
        valid_all = self._valid_all if self._valid_all is not None else (meas > 0)
        hole = rescued & ~valid_all
        self._plane_z = plane_z
        self._rescued = rescued if rescued.any() else None
        self._hole_rescued = hole if self._rescued is not None else None
        area = float(domain.sum())
        self.plane_info = {
            "enabled": True, "ok": True, "coef": fit["coef"],
            "inlier_ratio": fit["inlier_ratio"], "rms_mm": fit["rms_mm"],
            "n_fit": fit["n_fit"],
            "rescued_frac": float(rescued.sum()) / max(area, 1.0),
            "rescued_frame_frac": float(rescued.mean())}
        self._build_maps()

    def calibrate_touch(self, frames, margin_mm=15):
        """Set the contact-band ceiling from the player's own feet.

        The camera only ever sees the TOP of the shoe — a press reads as
        "shoe-top height above the surface", and that height is the unknown
        shoe thickness (bare foot ~20mm, sneaker ~50-60mm). While the player
        STANDS on the mat, contact is guaranteed, so the foot-top height
        measured now is exactly what a real press will read later. The band
        ceiling becomes that measurement + margin: thick shoes get the reach
        they need, bare feet get a much TIGHTER band than the default.

        ``frames`` are aligned depth frames of the player standing still on
        the captured surface. Returns the new ceiling in mm, or None if no
        foot-sized object was found (state is then unchanged).
        """
        samples = []
        m = self._maps
        for f in frames:
            d = f.astype(np.int32)
            above = m.surface - d
            valid = (d > 0) & (m.surface > 0) & (~m.artifact)
            obj = self._foot_object_mask(above, valid)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(
                obj.astype(np.uint8), connectivity=8)
            # Only the largest foot-sized objects: mat wrinkles left by earlier
            # presses form MANY small in-band objects that would outvote the
            # one real foot and drag the ceiling far below the shoe height.
            order = sorted(range(1, n),
                           key=lambda b: -int(stats[b, cv2.CC_STAT_AREA]))
            for b in order[:2]:   # at most two feet
                area = stats[b, cv2.CC_STAT_AREA]
                if not (1500 <= area <= 0.15 * obj.size):
                    continue
                vals = above[(lab == b) & valid & (above > FOOT_OBJ_MIN_MM)]
                vals = vals[vals < 150]   # foot region; drop the shin/leg
                if vals.size < 300:
                    continue
                # 70th percentile = the shoe's TOP surface (lower percentiles
                # are the sloped shoe edges, higher ones the ankle/shin)
                samples.append(float(np.percentile(vals, 70)))
        if len(samples) < 3:
            return None
        press_top = float(np.median(samples))
        self.contact_max = int(np.clip(press_top + margin_mm, 25, 120))
        self._build_maps()
        return self.contact_max

    def set_tiles(self, label_map, tile_ids):
        self.label_map = label_map
        self.tile_ids = list(tile_ids)
        lab = label_map[label_map >= 0]
        self._tile_area = np.bincount(lab, minlength=len(self.tile_ids)) \
            if lab.size else np.zeros(len(self.tile_ids), np.int64)
        self._reset_runtime()
        if self._calibrated and self.plane_fallback:
            mat = self._mat_mask_from_labels()
            old = self._mat_mask
            if (mat is None) != (old is None) or (
                    mat is not None and not np.array_equal(mat, old)):
                self._update_plane()

    def _reset_runtime(self):
        self._prev = None
        self._motion_age = None
        self._band_age = None
        self._streak = {}
        self._miss = {}
        self._quiet = {}
        self.held = set()

    # -- per-frame ------------------------------------------------------------

    def _contact_band(self, d, m=None):
        m = m or self._maps
        above = m.surface - d
        return ((d > 0) & (m.surface > 0) & (~m.artifact) &
                (above >= m.min_map) & (above <= m.max_map))

    def update(self, depth_mm):
        """Feed one aligned depth frame; returns the set of tiles newly pressed."""
        if depth_mm is None or self.label_map is None or not self.tile_ids:
            return set()
        d = depth_mm.astype(np.int32)
        m = self._maps     # one consistent snapshot for the whole frame

        # Motion gate: remember which pixels changed recently (vs their own
        # noise level). On a static scene nothing has motion -> nothing presses.
        if self._motion_age is None:
            self._motion_age = np.zeros(d.shape, np.int16)
        else:
            np.subtract(self._motion_age, 1, out=self._motion_age,
                        where=self._motion_age > 0)
        if self._prev is not None:
            moved = (d > 0) & (self._prev > 0) & \
                    (np.abs(d - self._prev) > m.motion_map)
            if m.hole_rescued is not None:
                # over a rescued hole the empty mat read 0: a foot appearing
                # there goes 0 -> valid, which is movement too
                moved |= m.hole_rescued & (self._prev == 0) & (d > 0)
            self._motion_age[moved] = self.motion_hold
        self._prev = d

        # Band stability: a pixel must STAY in the contact band for consecutive
        # frames. Noise flicker jumps in and out; a resting fingertip doesn't.
        band = self._contact_band(d, m)
        if self._band_age is None:
            self._band_age = np.zeros(d.shape, np.int16)
        self._band_age[~band] = 0
        self._band_age[band] += 1

        above = m.surface - d
        valid = (d > 0) & (m.surface > 0) & (~m.artifact)
        hover = valid & (above > m.max_map)
        # Presence: in-band pixels NEAR a hovering mass (the foot standing on
        # the key). No motion gate — a resting foot goes motion-silent but must
        # keep its tile held. No far-away band pixels either: a wrinkle the
        # press left in the soft mat has no foot above it and must neither
        # hold a key forever nor block re-arming.
        if hover.any():
            k = 2 * PRESENCE_RADIUS_PX + 1
            near_hover = cv2.dilate(hover.astype(np.uint8),
                                    np.ones((k, k), np.uint8)) > 0
            present_band = band & near_hover
        else:
            present_band = np.zeros_like(band)

        contact = band & (self._band_age >= self.band_stable) & (self._motion_age > 0)
        if contact.any():
            # the connected "foot object": foot/shin-height pixels — used to
            # recognise that several contact fragments belong to ONE foot
            foot_obj = self._foot_object_mask(above, valid)
            contact = self._ring_guard(contact, hover)
            candidates = (self._blob_tiles(contact, hover, foot_obj, present_band)
                          if contact.any() else set())
        else:
            candidates = set()
        return self._temporal(candidates, self._present_tiles(present_band))

    def _present_tiles(self, band):
        """Tiles that still have enough in-band pixels over them (foot there)."""
        labels = self.label_map[band]
        labels = labels[labels >= 0]
        if labels.size == 0:
            return set()
        counts = np.bincount(labels, minlength=len(self.tile_ids))
        return {self.tile_ids[k] for k in np.nonzero(counts >= PRESENT_MIN_PX)[0]}

    @staticmethod
    def _foot_object_mask(above, valid):
        """Connected mask of anything standing on the surface (foot/shin).

        Depth holes at shoe edges and near-surface (<5mm) gaps shatter the raw
        elevation mask into hundreds of fragments — the closing bridges them so
        one physical foot really is ONE component.
        """
        obj = (valid & (above > FOOT_OBJ_MIN_MM) &
               (above < FOOT_OBJ_MAX_MM)).astype(np.uint8)
        return cv2.morphologyEx(obj, cv2.MORPH_CLOSE,
                                np.ones((15, 15), np.uint8)) > 0

    @staticmethod
    def _ring_guard(contact, hover):
        """Drop contact pixels hugging a hovering mass (a foot's sloped edge).

        The sides of a foot standing NEXT to a key slice through the contact
        band as a thin ring around the hovering foot top — that ring must not
        press. A real press extends well beyond its limb's outline, so erasing
        the few px closest to hover costs it almost nothing.
        """
        if not hover.any():
            return contact
        k = 2 * RING_GUARD_PX + 1
        near_hover = cv2.dilate(hover.astype(np.uint8),
                                np.ones((k, k), np.uint8)) > 0
        return contact & ~near_hover

    def _blob_tiles(self, contact, hover, foot_obj=None, band=None):
        """Finger-sized connected blobs -> the one tile each blob covers most.

        ``hover`` marks pixels ABOVE the band (the finger/foot/leg the press
        belongs to). A contact blob with no hover pixels next to it is nothing
        standing on the surface — typically depth wobble under a moving shadow
        (sunlight IR interference) — and is rejected, unless it touches the
        frame border (the limb may be outside the view).

        ``foot_obj`` connects the fragments of one physical foot: a blob whose
        foot object already HOLDS a tile is the same foot rolling/spreading —
        it must not sound the neighbour keys as well.
        """
        mask = contact.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, blobs, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        h, w = mask.shape
        max_area = self.max_blob_frac * h * w

        # objects (feet) that currently hold a tile: their in-band pixels over
        # a held tile mark the object as "already sounding"
        holding_objs: set = set()
        obj_lab = None
        if foot_obj is not None:
            _n_obj, obj_lab = cv2.connectedComponents(
                foot_obj.astype(np.uint8), connectivity=8)
        if obj_lab is not None and self.held and band is not None:
            for k in range(len(self.tile_ids)):
                if self.tile_ids[k] not in self.held:
                    continue
                ids = obj_lab[band & (self.label_map == k)]
                ids = ids[ids > 0]
                if ids.size:
                    holding_objs.update(np.unique(ids).tolist())

        best_by_obj: dict = {}   # foot object -> (area, tile): ONE note per foot
        hits = set()
        for b in range(1, n):
            area = stats[b, cv2.CC_STAT_AREA]
            if area < self.min_blob_px or area > max_area:
                continue
            if self.require_attached and not self._is_attached(b, stats, hover, h, w):
                continue
            keys = self.label_map[blobs == b]
            keys = keys[keys >= 0]
            if keys.size == 0:
                continue
            counts = np.bincount(keys, minlength=len(self.tile_ids))
            # Same-foot guards: substantial overlap with a held tile, or the
            # blob's foot object already holds one — pressing F#4 must not
            # also sound F4 and E4 while the foot rolls flat.
            held_px = sum(int(counts[k]) for k in range(len(self.tile_ids))
                          if self.tile_ids[k] in self.held)
            if held_px >= HELD_OVERLAP_FRAC * area:
                continue
            blob_obj = None
            if obj_lab is not None:
                ids = obj_lab[blobs == b]
                ids = ids[ids > 0]
                if ids.size:
                    blob_obj = int(np.bincount(ids).argmax())
                    if blob_obj in holding_objs:
                        continue
            # The tile under the blob's CENTRE — the middle of the foot decides,
            # not whichever tile the rim happens to graze.
            cx, cy = centroids[b]
            tid = self.label_map[min(h - 1, max(0, int(round(cy)))),
                                 min(w - 1, max(0, int(round(cx))))]
            if tid < 0:  # centre off the keys (e.g. curved blob) -> majority vote
                tid = int(np.argmax(counts))
            tile = self.tile_ids[int(tid)]
            if blob_obj is None:
                hits.add(tile)
            else:
                # options for this foot: the centre tile (2x weight) plus every
                # tile the blob covers substantially — so a key already on a
                # confirm streak stays in the running even when the flattening
                # foot no longer forms a separate fragment on it
                opts = best_by_obj.setdefault(blob_obj, [])
                for k in np.nonzero(counts >= self.min_blob_px)[0]:
                    cover = float(counts[k]) / max(float(self._tile_area[k]), 1.0)
                    if int(k) == int(tid):
                        cover *= 2.0
                    opts.append((cover, self.tile_ids[int(k)]))
        # ONE new note per foot per frame (a foot landing across a key
        # boundary must not play a chord). Ranking: (1) a tile already on a
        # confirm streak keeps the candidacy — the toe touches F#4 first, then
        # the foot rolls flat and F4's area would overtake; first contact
        # wins. (2) otherwise the fragment COVERING its key best — a toe
        # covers the narrow black key far more than the foot's bulk covers
        # the wide white beside it.
        for options in best_by_obj.values():
            best = max(options,
                       key=lambda ct: (self._streak.get(ct[1], 0), ct[0]))
            hits.add(best[1])
        return hits

    @staticmethod
    def _is_attached(b, stats, hover, h, w):
        """True if blob b has above-band pixels around its bbox or touches the
        frame border (then the limb may simply be outside the view)."""
        x, y = stats[b, cv2.CC_STAT_LEFT], stats[b, cv2.CC_STAT_TOP]
        bw, bh = stats[b, cv2.CC_STAT_WIDTH], stats[b, cv2.CC_STAT_HEIGHT]
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            return True
        r = ATTACH_RADIUS_PX + RING_GUARD_PX  # ring guard pushed hover away
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + bw + r), min(h, y + bh + r)
        return int(hover[y0:y1, x0:x1].sum()) >= ATTACH_MIN_HOVER_PX

    def live_masks(self, depth_mm):
        """(band, contact, hover) for one aligned frame — read-only.

        ``contact`` includes the stability/motion/ring-guard gates, i.e. it is
        exactly the pixel set that can feed blobs. Safe to call from the
        streaming thread while update() runs on the capture thread.
        """
        d = depth_mm.astype(np.int32)
        m = self._maps
        above = m.surface - d
        hover = ((d > 0) & (m.surface > 0) & (~m.artifact) &
                 (above > m.max_map))
        band = self._contact_band(d, m)
        if self._band_age is not None and self._motion_age is not None:
            contact = band & (self._band_age >= self.band_stable) & \
                      (self._motion_age > 0)
            contact = self._ring_guard(contact, hover)
        else:
            contact = np.zeros_like(band)
        return band, contact, hover

    def render_debug(self, depth_mm):
        """BGR visualization of one aligned frame, OrbbecViewer-style.

        Base: TURBO distance colormap (near = blue, far = red) like the
        official viewer. Overlays: yellow = in the contact band, red = pixels
        that can actually fire (band + stable + motion + ring guard), cyan
        outline = hovering object (foot/hand above the band), darkened =
        masked artifact pixels, magenta tint = reference from the fitted plane
        (plane fallback), near-black = no depth reading.
        """
        d = depth_mm.astype(np.float32)
        m = self._maps
        valid = (depth_mm > 0) & (m.surface > 0)
        surf = m.surface[m.surface > 0]
        lo = max(200.0, float(np.percentile(surf, 5)) - 600.0) if surf.size else 400.0
        hi = float(np.percentile(surf, 95)) + 200.0 if surf.size else 3000.0
        norm = np.clip((d - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[depth_mm == 0] = (15, 15, 15)
        img[m.artifact & valid] = (img[m.artifact & valid] * 0.35).astype(np.uint8)
        if m.rescued is not None:   # magenta tint: reference comes from the plane
            r = m.rescued & valid
            img[r] = (0.5 * img[r] + 0.5 * np.float32((255, 0, 255))).astype(np.uint8)
        band, contact, hover = self.live_masks(depth_mm)
        if hover.any():   # outline the hovering object, keep its colormap fill
            cnts, _ = cv2.findContours(hover.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 0), 2)
        img[band] = (0, 220, 255)            # yellow: in the contact band
        img[contact] = (0, 0, 255)           # red: can fire
        return img


    def _temporal(self, candidates, present=frozenset()):
        """Confirm-frames + hold-while-present + re-arm quiet period per tile.

        ``candidates`` (motion-gated blobs) can only START a press. Whether a
        held press ENDS is decided by ``present`` — the foot's in-band pixels
        without the motion gate — so a foot resting on its key keeps the tile
        held even after the motion gate goes silent, and the lift-off pass
        back through the band cannot fire it a second time. Re-arming also
        waits until the area is actually clear.
        """
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
                    if tid in present:
                        self._miss[tid] = 0      # foot still on the key
                    else:
                        self._miss[tid] = self._miss.get(tid, 0) + 1
                        if self._miss[tid] >= self.release_frames:
                            self.held.discard(tid)
                            self._quiet[tid] = 0
                elif (tid in present and
                      self._quiet.get(tid, self.rearm_frames) < self.rearm_frames):
                    # released, but the foot still lingers over the key: the
                    # quiet countdown only runs once the area is clear. An
                    # ALREADY armed tile is not disarmed by presence — that
                    # would be the next press announcing itself.
                    self._quiet[tid] = 0
                else:
                    self._quiet[tid] = self._quiet.get(tid, self.rearm_frames) + 1
        return fired


def _hex_bgr(hex_color, fallback=(217, 144, 74)):
    """'#RRGGBB' -> BGR tuple (fallback on anything unparsable)."""
    try:
        s = str(hex_color).lstrip("#")
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return fallback


def render_semantic(tiles, width, height, hover=None, contact=None, held=()):
    """Semantic scene view: every tile as a colored region, the foot (hover)
    as an orange object, its contact patch red, held (sounding) tiles lit.

    Pure drawing — cheap enough to run per streamed frame (10 FPS).
    """
    img = np.full((height, width, 3), (26, 24, 22), np.uint8)
    held = set(held)
    for t in tiles:
        poly = t.get("polygon") or []
        if not t.get("enabled", True) or len(poly) < 3:
            continue
        pts = np.asarray(poly, np.int32)
        color = _hex_bgr(t.get("color"))
        is_held = t.get("id") in held
        fill = tuple(int(c * (0.85 if is_held else 0.30)) for c in color)
        cv2.fillPoly(img, [pts], fill)
        cv2.polylines(img, [pts], True, (255, 255, 255) if is_held else color,
                      2 if is_held else 1)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        cv2.putText(img, str(t.get("label", "")), (int(cx) - 10, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255) if is_held else (190, 190, 190), 1, cv2.LINE_AA)
    if hover is not None and hover.any():
        # the foot/limb as a translucent orange object
        obj = img.copy()
        obj[hover] = (30, 130, 255)
        img = cv2.addWeighted(obj, 0.65, img, 0.35, 0)
    if contact is not None and contact.any():
        img[contact] = (0, 0, 255)
        n, _blobs, stats, centroids = cv2.connectedComponentsWithStats(
            contact.astype(np.uint8), connectivity=8)
        for b in range(1, n):
            if stats[b, cv2.CC_STAT_AREA] < 20:
                continue
            cx, cy = centroids[b]
            cv2.circle(img, (int(cx), int(cy)), 4, (255, 255, 255), -1)
    return img


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
