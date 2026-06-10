"""Calibrate the key grid directly from the printed piano mat — no ArUco needed.

The mat itself is a calibration pattern: a bright rectangle on a darker floor,
with black keys printed in a known pattern (build_keyboard). That gives us
everything a marker-based calibration gives us, plus orientation:

  1. ``find_mat_quad``      — the mat's 4 corners (bright blob vs the floor);
  2. ``orient_corners``     — which corner is which: try the 4 flip hypotheses
                              and keep the one whose warped dark pixels best
                              overlap the model's black-key rectangles;
  3. ``refine_corners``     — measure the painted black-key centres in the
                              warped image, fit a linear x-correction against
                              the model centres, fold it back into the corners.

``auto_calibrate`` chains the three. Works on any BGR frame without a foot in
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
    """
    warped, _ = _warp(bgr, corners, width, height)
    dark = _dark_mask(warped)
    model = black_key_model(num_white, width, height)
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
    best_key, best_iou = None, -1.0
    for fkey, view in views.items():
        inter = (view & model).sum()
        union = (view | model).sum()
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_key, best_iou = fkey, iou
    return candidates[best_key], float(best_iou)


def measure_black_centers(warped_bgr, num_white, band=(0.2, 0.5), min_frac=0.8):
    """X-centres of the painted black keys, measured in a horizontal band.

    In rows 20–50%% of the canvas only black keys and the thin white-key
    separator LINES are dark; runs of dark columns narrower than a quarter
    white-key are lines (or noise) and are dropped. ``min_frac`` is how dark a
    column must be across the band — lower it for blurry low-res sources.
    """
    h, w = warped_bgr.shape[:2]
    dark = _dark_mask(warped_bgr)
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
                   width=constants.TARGET_WIDTH, height=constants.TARGET_HEIGHT):
    """Full mat-based calibration of one (foot-free) BGR frame.

    Returns (corners, info): corners as a float32 (4,2) array in TL,TR,BR,BL
    canvas order, or (None, info) when no mat-like region is found. info
    carries quality metrics: black-key IoU and the refinement residuals (px).
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    quad = find_mat_quad(bgr)
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
    if iou < 0.2:
        log.warning("mat orientation is uncertain (black-key IoU %.2f) — "
                    "check the key grid visually.", iou)
        info["weak"] = True
    return corners, info
