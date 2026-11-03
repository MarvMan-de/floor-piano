"""Calibrate the key grid directly from the printed piano mat — no ArUco needed.

The mat itself is a calibration pattern: a bright rectangle on a darker floor,
with black keys printed in a known pattern (build_keyboard). That gives us
everything a marker-based calibration gives us, plus orientation:

  1. ``quad_from_white_keys`` — PRIMARY: the mat's own white keys are isolated
                              bright quads (each is outlined by dark separator
                              lines). Find them, RANSAC a line through their
                              centroids, match them 1:1 to the model keys and
                              fit a homography -> mat corners. Robust against
                              bright backgrounds (white walls / furniture) that
                              merge with the mat in a plain brightness mask;
     ``find_mat_quad``      — FALLBACK: largest bright blob vs the floor;
  2. ``orient_corners``     — which corner is which: try the 4 flip hypotheses
                              and keep the one whose warped dark pixels best
                              overlap the model's black-key rectangles;
  3. ``refine_corners``     — measure the painted black-key centres in the
                              warped image, fit a linear x-correction against
                              the model centres, fold it back into the corners.

``auto_calibrate`` chains them. Works on any BGR frame without a foot in
it — pass a median background (``median_background``) for robustness.

Used by demo_video.py (so a plain clip of the mat needs no manual corners) and
as the ArUco fallback in calibrate.py (no printed markers required on site).
"""

import logging

import cv2
import numpy as np

import constants
from detection import build_keyboard

log = logging.getLogger(__name__)


def median_background(frames):
    """Per-pixel median of BGR frames — removes anything that moves (feet)."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def find_mat_quad(bgr, min_area_frac=0.2):
    """Return the mat's 4 corner points (float32, cyclic order), or None.

    The mat is the largest bright region in the frame (white vinyl vs floor).
    Returns None when no bright region covers at least ``min_area_frac`` of
    the frame — e.g. mat not in view, or a floor as bright as the mat.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 12:
        return None  # featureless frame — Otsu would "find" the whole image
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area_frac * gray.shape[0] * gray.shape[1]:
        return None
    peri = cv2.arcLength(biggest, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(biggest, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # Not clean enough for a 4-point polygon — take the min-area rectangle.
    return cv2.boxPoints(cv2.minAreaRect(biggest)).astype(np.float32)


# ── white-key pattern detection (primary quad source) ────────────────────────
#
# A plain brightness mask fails as soon as a white wall / cupboard touches the
# mat in the image: mat + background merge into one blob and its bounding quad
# is garbage. But the printed mat has a much stronger signature: every WHITE KEY
# is an isolated bright quad (separated by printed dark lines), all of similar
# size, with collinear centroids and known model geometry. Matching those blobs
# 1:1 to the model keys gives a full homography — and with it the mat corners —
# without ever segmenting the mat outline itself.

_model_pts_cache = {}


def white_key_model_points(num_white, width, height):
    """Per white key: (top-mid, centroid, bottom-mid) of its VISIBLE region.

    "Visible" = the key rectangle minus the black keys that overlap it — the
    same shape the camera sees as one bright blob. top-mid is the middle of the
    widest gap the key shows at the mat's top edge (between black keys);
    bottom-mid is the middle of its full width at the bottom edge.
    """
    key = (num_white, width, height)
    if key in _model_pts_cache:
        return _model_pts_cache[key]
    keys = build_keyboard(num_white, width, height)
    whites = [k for k in keys if k.kind == "white"]
    blacks = [k for k in keys if k.kind == "black"]
    pts = []
    for wk in whites:
        segs = [(wk.x0, wk.x1)]
        for bk in blacks:
            nxt = []
            for a, b in segs:
                if bk.x1 <= a or bk.x0 >= b:
                    nxt.append((a, b))
                    continue
                if bk.x0 > a:
                    nxt.append((a, bk.x0))
                if bk.x1 < b:
                    nxt.append((bk.x1, b))
            segs = nxt
        a, b = max(segs, key=lambda s: s[1] - s[0])
        mask = np.zeros((height, width), np.uint8)
        mask[wk.y0:wk.y1, wk.x0:wk.x1] = 1
        for bk in blacks:
            mask[bk.y0:bk.y1, bk.x0:bk.x1] = 0
        ys, xs = np.nonzero(mask)
        pts.append((((a + b) / 2.0, 0.0),
                    (float(xs.mean()), float(ys.mean())),
                    ((wk.x0 + wk.x1) / 2.0, float(height))))
    _model_pts_cache[key] = pts
    return pts


def find_white_key_blobs(bgr, max_cands=40):
    """Bright blobs that could be single white keys (contour, centroid, area).

    Otsu-bright external contours, filtered to key-plausible sizes and shapes.
    Large blobs (a whole wall, the merged mat outline) and speckles fall out via
    the area window; wildly elongated or hollow shapes via aspect / rect-fill.
    """
    h, w = bgr.shape[:2]
    frame_area = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 12:
        return []
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # RETR_CCOMP: outer boundary of EVERY bright component at the top level —
    # RETR_EXTERNAL would hide keys nested inside a hole of a white mat margin.
    cnts, hier = cv2.findContours(bright, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for ci, c in enumerate(cnts):
        if hier is not None and hier[0][ci][3] != -1:
            continue  # hole boundary (e.g. a black key inside the mat) — skip
        area = cv2.contourArea(c)
        # up to ~8% of the frame per key: a mat filling the whole view still
        # passes, a merged mat-plus-wall blob (tens of %) never does
        if not (0.001 * frame_area < area < 0.08 * frame_area):
            continue
        (cx, cy), (rw, rh), _ang = cv2.minAreaRect(c)
        long_, short = max(rw, rh), min(rw, rh)
        if short <= 0 or long_ / short > 8:
            continue
        if area / max(long_ * short, 1.0) < 0.5:
            continue  # very non-rectangular (L-shapes of merged background)
        cands.append({"contour": c, "cx": float(cx), "cy": float(cy),
                      "area": float(area)})
    cands.sort(key=lambda d: -d["area"])
    return cands[:max_cands]


def _collinear_key_cluster(cands, min_inliers, tol_frac=0.35):
    """Largest subset of blobs with collinear centroids and similar areas.

    Pair-RANSAC over centroid lines; tolerance scales with the typical key size.
    Returns (indices sorted along the line, unit direction) or (None, None).
    """
    n = len(cands)
    if n < min_inliers:
        return None, None
    pts = np.array([[c["cx"], c["cy"]] for c in cands])
    areas = np.array([c["area"] for c in cands])
    best_inl, best_u = None, None
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            length = float(np.hypot(d[0], d[1]))
            if length < 1e-6:
                continue
            u = d / length
            nvec = np.array([-u[1], u[0]])
            dist = np.abs((pts - pts[i]) @ nvec)
            tol = tol_frac * np.sqrt(float(np.median(areas)))
            inl = np.where(dist < tol)[0]
            if len(inl) >= min_inliers:
                med = float(np.median(areas[inl]))
                inl = np.array([k for k in inl
                                if 0.35 * med < areas[k] < 2.8 * med])
            if len(inl) >= min_inliers and (
                    best_inl is None or len(inl) > len(best_inl)):
                best_inl, best_u = inl, u
    if best_inl is None:
        return None, None
    order = np.argsort(pts[best_inl] @ best_u)
    return best_inl[order], best_u


def _blob_extreme_mids(contour, u):
    """(top-mid, bottom-mid, centroid) of a blob across the keyboard line.

    'top'/'bottom' = the extreme 12% bands of the contour along the direction
    perpendicular to the keyboard line u; their mean approximates the middle of
    the key's edge segment at the mat border (matches white_key_model_points).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    nvec = np.array([-u[1], u[0]])
    s = pts @ nvec
    lo, hi = float(s.min()), float(s.max())
    band = 0.12 * (hi - lo)
    top = pts[s <= lo + band].mean(axis=0)
    bot = pts[s >= hi - band].mean(axis=0)
    return top, bot, pts.mean(axis=0)


def quad_from_white_keys(bgr, num_white=None,
                         width=constants.TARGET_WIDTH,
                         height=constants.TARGET_HEIGHT):
    """Mat corners from the white-key pattern, or None.

    Finds the white-key blobs, fits model->image homographies over the possible
    contiguous key windows (handles a few keys hidden or an extra bright blob on
    the line) and keeps the candidate whose warp best matches the model's black
    keys. Orientation is NOT resolved here (a mirrored mat fits a mirrored
    homography just as well) — auto_calibrate's orient_corners does that.
    Returns cyclic float32 (4, 2) corners.
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    min_inliers = max(4, num_white - 4)
    cands = find_white_key_blobs(bgr)
    idx, u = _collinear_key_cluster(cands, min_inliers)
    if idx is None:
        return None
    keys = [cands[i] for i in idx]
    model = white_key_model_points(num_white, width, height)
    measured = [_blob_extreme_mids(k["contour"], u) for k in keys]

    h_img, w_img = bgr.shape[:2]
    diag = float(np.hypot(w_img, h_img))
    m = min(len(keys), num_white)
    canvas = _canvas_dst(width, height).reshape(-1, 1, 2)
    best_quad, best_iou = None, -1.0
    # All contiguous windows: measured (may contain an extra blob) x model (a
    # few keys may be occluded at either end). Ambiguity between windows is
    # settled by the warped black-key IoU, not by reprojection error alone.
    for mo in range(len(keys) - m + 1):
        for ko in range(num_white - m + 1):
            src, dst = [], []
            for (mt, mc, mb), (top, bot, cen) in zip(
                    model[ko:ko + m], measured[mo:mo + m]):
                src += [mt, mc, mb]
                dst += [top.tolist(), cen.tolist(), bot.tolist()]
            H, inl_mask = cv2.findHomography(
                np.float32(src), np.float32(dst), cv2.RANSAC,
                max(3.0, 0.012 * diag))
            # Glare/shadows nibble at the blobs, so centroid points scatter —
            # accept half the correspondences as inliers; the warped black-key
            # IoU below is the real quality gate.
            if H is None or inl_mask is None or inl_mask.sum() < 1.5 * m:
                continue
            quad = cv2.perspectiveTransform(canvas, H).reshape(4, 2)
            if not np.isfinite(quad).all():
                continue
            if not cv2.isContourConvex(quad.astype(np.float32).round().astype(np.int32)):
                continue
            area = cv2.contourArea(quad.astype(np.float32))
            if not (0.01 * w_img * h_img < area < 4.0 * w_img * h_img):
                continue
            _, iou = orient_corners(bgr, order_long_axis(np.float32(quad)),
                                    num_white, width, height)
            if iou > best_iou:
                best_quad, best_iou = np.float32(quad), iou
    if best_quad is None or best_iou < 0.2:
        return None
    return best_quad


def order_long_axis(quad):
    """Order 4 cyclic points as [TL, TR, BR, BL] with the LONG side horizontal.

    The keyboard runs along the mat's long axis, so the long side must become
    the canvas width. Which end is C4 / which side the black keys are on is
    deliberately left open — orient_corners resolves that afterwards.
    """
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    quad = quad[np.argsort(ang)]
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    si = int(np.argmin(edges))  # short edge quad[si] -> quad[si+1] = canvas LEFT side
    tl, bl = quad[(si + 1) % 4], quad[si]
    tr, br = quad[(si + 2) % 4], quad[(si + 3) % 4]
    return np.float32([tl, tr, br, bl])


def _canvas_dst(width, height):
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]])


def _warp(bgr, corners, width, height):
    M = cv2.getPerspectiveTransform(np.float32(corners), _canvas_dst(width, height))
    return cv2.warpPerspective(bgr, M, (width, height)), M


def _dark_mask(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return dark > 0


def black_key_model(num_white, width, height, start_octave=None):
    """Boolean canvas mask of where the model expects the black keys."""
    mask = np.zeros((height, width), bool)
    for k in build_keyboard(num_white, width, height, start_octave):
        if k.kind == "black":
            mask[k.y0:k.y1, k.x0:k.x1] = True
    return mask


def orient_corners(bgr, corners, num_white,
                   width=constants.TARGET_WIDTH, height=constants.TARGET_HEIGHT):
    """Resolve the 4 flip hypotheses; returns (corners, iou).

    Flipping the warped image horizontally is the same as swapping which source
    corners map to the canvas's left vs right side (and vertically: top vs
    bottom) — so the winning flip is folded back into the corner ORDER and the
    plain TL,TR,BR,BL warp in main.py needs no extra flip logic.

    Scoring, per flip candidate: (1) how many key-shaped dark BARS match the
    model centres 1:1 — bars are measured in an ADAPTIVE row band (the rows
    that actually contain key-wide dark runs), because on a roughly placed
    user quad the black keys don't sit at the model's fixed rows; runs wider
    than a key (mat-edge shadow/trim) never count. (2) that band must lie
    near the canvas TOP, where the model puts the black keys. (3) raw dark
    IoU as final tie-breaker — and the sole criterion when no candidate sees
    a bar pattern at all (blurry/low-res sources).
    """
    warped, _ = _warp(bgr, corners, width, height)
    dark = _dark_mask(warped)
    model = black_key_model(num_white, width, height)
    model_ctr = model_black_centers(num_white, width, height)
    white_w = width / num_white
    tl, tr, br, bl = corners
    candidates = {
        (False, False): corners,
        (True, False): np.float32([tr, tl, bl, br]),   # mirror left-right
        (False, True): np.float32([bl, br, tr, tl]),   # mirror top-bottom
        (True, True): np.float32([br, bl, tl, tr]),    # 180 degrees
    }
    views = {
        (False, False): dark,
        (True, False): dark[:, ::-1],
        (False, True): dark[::-1, :],
        (True, True): dark[::-1, ::-1],
    }
    scores = {}
    for fkey, view in views.items():
        inter = (view & model).sum()
        union = (view | model).sum()
        iou = inter / max(union, 1)
        centers, band_center = _adaptive_bar_centers(view, num_white)
        bars = len(_match_centers(centers, model_ctr, white_w))
        # band nearer the top is better: model black keys hang from y=0
        top_score = -band_center if band_center is not None else -1.0
        scores[fkey] = (bars, top_score, iou)
    if max(s[0] for s in scores.values()) >= 4:
        best_key = max(scores, key=lambda k: scores[k])
    else:
        best_key = max(scores, key=lambda k: scores[k][2])
    return candidates[best_key], float(scores[best_key][2])


def _adaptive_bar_centers(dark, num_white):
    """Black-key bar x-centres from the rows that actually contain key-wide
    dark runs. Returns (centers, band_center in 0..1) or ([], None).

    Unlike the fixed 20–50%% band of measure_black_centers, this finds the
    black keys wherever a rough quad put them vertically. Rows qualify when
    they hold at least 3 dark runs of key-like width (thin separator lines and
    full-width shadows both fail that test).
    """
    h, w = dark.shape[:2]
    white_w = w / num_white
    lo, hi = 0.25 * white_w, 1.2 * white_w
    counts = np.zeros(h, dtype=np.int32)
    padded = np.zeros(w + 2, dtype=np.int8)
    for y in range(h):
        padded[1:-1] = dark[y]
        edges = np.flatnonzero(np.diff(padded))
        runs = edges[1::2] - edges[::2]
        counts[y] = int(np.sum((runs > lo) & (runs < hi)))
    if counts.max() < 3:
        return [], None
    band = counts >= max(3, int(0.5 * counts.max()))
    band_rows = np.flatnonzero(band)
    col_frac = dark[band].mean(axis=0)
    is_bar = col_frac > 0.5
    centers = []
    x = 0
    while x < w:
        if is_bar[x]:
            x0 = x
            while x < w and is_bar[x]:
                x += 1
            run = x - x0
            if lo < run < hi:
                centers.append(x0 + run / 2.0)
        else:
            x += 1
    return centers, float(band_rows.mean()) / max(h - 1, 1)


def measure_black_centers(warped_bgr, num_white, band=(0.2, 0.5), min_frac=0.8):
    """X-centres of the painted black keys, measured in a horizontal band.

    In rows 20–50%% of the canvas only black keys and the thin white-key
    separator LINES are dark; runs of dark columns narrower than a quarter
    white-key are lines (or noise) and are dropped. ``min_frac`` is how dark a
    column must be across the band — lower it for blurry low-res sources.
    """
    return _bar_centers_from_dark(_dark_mask(warped_bgr), num_white, band, min_frac)


def _bar_centers_from_dark(dark, num_white, band=(0.2, 0.5), min_frac=0.8):
    """measure_black_centers on an already-computed boolean dark mask."""
    h, w = dark.shape[:2]
    rows = dark[int(band[0] * h):int(band[1] * h)]
    col_frac = rows.mean(axis=0)
    is_bar = col_frac > min_frac
    white_w = w / num_white
    centers = []
    x = 0
    while x < w:
        if is_bar[x]:
            x0 = x
            while x < w and is_bar[x]:
                x += 1
            run = x - x0
            if 0.25 * white_w < run < 1.2 * white_w:
                centers.append(x0 + run / 2.0)
        else:
            x += 1
    return centers


def model_black_centers(num_white, width, height, start_octave=None):
    return [(k.x0 + k.x1) / 2.0
            for k in build_keyboard(num_white, width, height, start_octave)
            if k.kind == "black"]


def _match_centers(measured, model, white_w):
    """1:1 match measured bars to model centres within half a key; (model, measured) pairs."""
    pairs = []
    used = set()
    for mx in measured:
        best_j, best_d = None, 0.5 * white_w
        for j, cx in enumerate(model):
            d = abs(mx - cx)
            if j not in used and d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            used.add(best_j)
            pairs.append((model[best_j], mx))
    return pairs


def refine_corners(bgr, corners, num_white,
                   width=constants.TARGET_WIDTH, height=constants.TARGET_HEIGHT,
                   min_matches=6):
    """Linear x-correction of the warp so painted black keys land on model centres.

    Fits measured = b*model + a over the matched bar centres and composes the
    inverse with the homography, then reads the corrected source corners back
    out. Returns (corners, info) — original corners when no safe fit exists.
    """
    warped, M = _warp(bgr, corners, width, height)
    white_w = width / num_white
    model = model_black_centers(num_white, width, height)
    # Strict first; relax for blurry low-res sources where the bars don't
    # reach 80% darkness across the measurement band.
    pairs, used_frac = [], 0.8
    for used_frac in (0.8, 0.65, 0.5):
        pairs = _match_centers(measure_black_centers(warped, num_white, min_frac=used_frac),
                               model, white_w)
        if len(pairs) >= min_matches:
            break
    info = {"matched_bars": len(pairs), "refined": False,
            "residual_before": None, "residual_after": None}
    if len(pairs) < min_matches:
        return corners, info
    mx = np.array([p[0] for p in pairs])
    px = np.array([p[1] for p in pairs])
    info["residual_before"] = float(np.mean(np.abs(px - mx)))
    b, a = np.polyfit(mx, px, 1)
    if not (0.85 < b < 1.15 and abs(a) < 0.15 * width):
        log.warning("mat refinement fit out of safe range (b=%.3f a=%.1f) — skipped", b, a)
        return corners, info
    # painted = b*model + a  ->  canvas correction x' = (x - a) / b
    C = np.array([[1.0 / b, 0.0, -a / b], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    M_new = C @ M
    src = cv2.perspectiveTransform(_canvas_dst(width, height).reshape(-1, 1, 2),
                                   np.linalg.inv(M_new)).reshape(4, 2)
    new_corners = np.float32(src)
    # re-measure to report the achieved residual
    warped2, _ = _warp(bgr, new_corners, width, height)
    pairs2 = _match_centers(measure_black_centers(warped2, num_white, min_frac=used_frac),
                            model, white_w)
    if pairs2:
        info["residual_after"] = float(np.mean(np.abs(
            np.array([p[1] for p in pairs2]) - np.array([p[0] for p in pairs2]))))
    info["refined"] = True
    return new_corners, info


def auto_calibrate(bgr, num_white=None,
                   width=constants.TARGET_WIDTH, height=constants.TARGET_HEIGHT,
                   min_area_frac=0.2):
    """Full mat-based calibration of one (foot-free) BGR frame.

    Returns (corners, info): corners as a float32 (4,2) array in TL,TR,BR,BL
    canvas order, or (None, info) when no mat-like region is found. info
    carries quality metrics: black-key IoU and the refinement residuals (px).

    ``min_area_frac`` is the smallest fraction of the frame the mat may cover.
    The default (0.2) suits a mat that fills the view (e.g. the test clips); pass
    a smaller value when the mat is only a strip of a wider camera view.
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    # Primary: the white-key pattern (robust against bright walls/furniture
    # merging with the mat). Fallback: the classic biggest-bright-blob quad.
    method = "white_keys"
    quad = quad_from_white_keys(bgr, num_white, width, height)
    if quad is None:
        method = "bright_region"
        quad = find_mat_quad(bgr, min_area_frac)
    if quad is None:
        return None, {"reason": "no mat-sized bright region found"}
    corners = order_long_axis(quad)
    corners, iou = orient_corners(bgr, corners, num_white, width, height)
    if iou < 0.05:
        # A bright region with no black-key pattern at all is not a piano mat
        # (a plain rug, a sheet of paper, an empty bright floor).
        return None, {"reason": f"bright region has no black-key pattern (IoU {iou:.2f})"}
    corners, info = refine_corners(bgr, corners, num_white, width, height)
    info["black_key_iou"] = iou
    info["method"] = method
    if iou < 0.2:
        log.warning("mat orientation is uncertain (black-key IoU %.2f) — "
                    "check the key grid visually.", iou)
        info["weak"] = True
    return corners, info
