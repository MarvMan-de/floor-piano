from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np

# The robust mat detector + keyboard geometry live in ../src (the depth-pipeline
# code). Put it on the path so we can reuse them instead of re-inventing detection.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import constants  # noqa: E402
import mat_calibration  # noqa: E402
from detection import build_keyboard  # noqa: E402

from webui.tile_store import MAX_TILES


def _content_roi(frame: np.ndarray) -> "tuple[np.ndarray, int, int]":
    """Strip pure-black letterbox bars. Returns (content, x_offset, y_offset).

    The colour source pads the frame with black bars to keep aspect ratio; those
    bars would otherwise be the biggest 'bright' rectangle and get mistaken for
    the mat. Detection must run on the real content only.
    """
    nonblack = frame.any(axis=2)
    rows = np.where(nonblack.any(axis=1))[0]
    cols = np.where(nonblack.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return frame, 0, 0
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return frame[y0:y1, x0:x1], x0, y0


def tiles_from_corners(
    corners,
    num_white: "int | None" = None,
    max_tiles: int = MAX_TILES,
) -> list[dict[str, Any]]:
    """Project the known keyboard grid into a 4-corner mat quad.

    ``corners`` are the mat's 4 corners in image pixels, TL,TR,BR,BL. The key grid
    is defined on a fixed canvas; projecting it back through the perspective
    transform makes every tile an actual piano key (white + black) with the right
    note name — and, crucially, perspective-correct: a tilted camera yields
    trapezoidal key polygons that follow the mat. This is the single source of
    truth for turning corners into tiles (manual placement OR auto-detected).
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("corners must be 4 (x, y) points in TL,TR,BR,BL order")

    w, h = constants.TARGET_WIDTH, constants.TARGET_HEIGHT
    canvas = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    to_image = cv2.getPerspectiveTransform(canvas, corners)  # canvas -> image

    keys = build_keyboard(num_white, w, h)
    blacks = [k for k in keys if k.kind == "black"]

    tiles: list[dict[str, Any]] = []
    for key in keys[:max_tiles]:
        canvas_poly = _key_canvas_polygon(key, blacks)
        pts = cv2.perspectiveTransform(
            np.float32([canvas_poly]), to_image)[0]
        polygon = [[int(round(px)), int(round(py))] for px, py in pts]
        tiles.append({"polygon": polygon, "label": key.name, "note": key.name})
    return tiles


def _key_canvas_polygon(key, blacks) -> "list[list[float]]":
    """The key's polygon on the flat canvas — real piano-key shapes.

    Black keys stay rectangles. White keys get the classic NOTCHED shape (the
    upper part narrowed where a black key sits), so white and black tiles no
    longer overlap: a press on F#4 lands ONLY on F#4, not visually/logically
    on the whites beneath it as well.
    """
    if key.kind == "black":
        return [[key.x0, key.y0], [key.x1, key.y0],
                [key.x1, key.y1], [key.x0, key.y1]]
    left = next((b for b in blacks if b.x0 < key.x0 < b.x1), None)
    right = next((b for b in blacks if b.x0 < key.x1 < b.x1), None)
    tl = float(left.x1) if left else float(key.x0)
    tr = float(right.x0) if right else float(key.x1)
    bh = float(blacks[0].y1) if blacks else float(key.y0)
    x0, x1 = float(key.x0), float(key.x1)
    y0, y1 = float(key.y0), float(key.y1)
    if left and right:
        return [[tl, y0], [tr, y0], [tr, bh], [x1, bh],
                [x1, y1], [x0, y1], [x0, bh], [tl, bh]]
    if right:
        return [[x0, y0], [tr, y0], [tr, bh], [x1, bh], [x1, y1], [x0, y1]]
    if left:
        return [[tl, y0], [x1, y0], [x1, y1], [x0, y1], [x0, bh], [tl, bh]]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def order_corners(corners, frame_bgr: "np.ndarray | None" = None,
                  num_white: "int | None" = None) -> "tuple[np.ndarray, str]":
    """4 mat corners in ANY click order -> TL,TR,BR,BL, oriented to the mat.

    The UI lets the user click/drag the 4 corners in any order; a naive
    sum/diff sort mislabels oblique quads (one point can 'win' two roles, which
    collapses the perspective transform and with it all 24 keys). This instead:

    1. validates the quad (4 distinct points, non-degenerate area, convex),
    2. sorts the points cyclically and puts the LONG side horizontal
       (the keyboard runs along the mat's long axis),
    3. when a camera frame is given, resolves the 4 flip hypotheses against the
       printed black-key pattern (mat_calibration.orient_corners) so C4 really
       is C4; without a frame — or when the mat shows no pattern, e.g. lying
       KEYS-DOWN with its blank back up — it falls back to a screen-geometric
       orientation (the user then fixes C4 side / key edge with the flip
       buttons, folded in server-side via flip_corners()).

    Raises ValueError for degenerate input. Returns (float32 (4, 2), how),
    where how is "pattern" (black keys decided) or "geometry" (fallback).
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("corners must be 4 finite (x, y) points")
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < 2.0:
                raise ValueError("two corners coincide — drag them apart")

    ordered = mat_calibration.order_long_axis(pts)  # cyclic sort + long side horizontal
    quad_i32 = ordered.round().astype(np.int32)
    if cv2.contourArea(ordered) < 64.0 or not cv2.isContourConvex(quad_i32):
        raise ValueError("corners form a degenerate (crossed or collapsed) quad")

    if frame_bgr is not None:
        oriented, iou = mat_calibration.orient_corners(
            frame_bgr, ordered, num_white)
        if iou >= 0.05:  # only trust the pattern when it actually saw black keys
            return oriented, "pattern"

    # Geometric fallback: top edge = the long side that is higher on screen,
    # left = smaller x. Matches what a user looking at the image expects.
    tl, tr, br, bl = ordered
    if tl[1] + tr[1] > bl[1] + br[1]:
        tl, tr, br, bl = bl, br, tr, tl      # vertical mirror
    if tl[0] > tr[0]:
        tl, tr, br, bl = tr, tl, bl, br      # horizontal mirror
    return np.float32([tl, tr, br, bl]), "geometry"


def flip_corners(ordered, flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    """User-requested orientation override on TL,TR,BR,BL corners.

    Needed when the mat lies keys-down (blank back up): no printed pattern is
    visible, so which end is C4 (``flip_h``) and along which long edge the
    black keys run (``flip_v``) cannot be detected — the user says so instead.
    Also lets the user overrule a wrong pattern detection.
    """
    tl, tr, br, bl = np.asarray(ordered, dtype=np.float32)
    if flip_h:
        tl, tr, br, bl = tr, tl, bl, br      # swap the C4 side
    if flip_v:
        tl, tr, br, bl = bl, br, tr, tl      # black keys along the other edge
    return np.float32([tl, tr, br, bl])


def detect_corners(frame_bgr: np.ndarray, num_white: "int | None" = None):
    """Best-effort auto-detect of the mat's 4 corners (TL,TR,BR,BL), or None.

    Reuses src/mat_calibration.auto_calibrate (finds the printed mat, orients and
    sub-pixel-refines it from the black-key pattern). Runs on the real image
    content, not the black letterbox bars. This is only a *starting point* for the
    corner tool — brightness-based detection is scene-dependent, so the user drags
    the corners to fix them.
    """
    num_white = constants.DEFAULT_NUM_WHITE if num_white is None else num_white
    content, x_off, y_off = _content_roi(frame_bgr)
    try:
        # The mat is usually only a wide strip of the camera view, so accept a
        # much smaller bright region than the whole frame.
        corners, _info = mat_calibration.auto_calibrate(
            content, num_white, min_area_frac=0.05)
    except Exception:
        return None
    if corners is None:
        return None
    corners = np.asarray(corners, dtype=np.float32) + [x_off, y_off]  # -> full-frame coords
    # The refined quad may extrapolate slightly beyond the frame (a mat edge
    # cut off at the border). Clamp into the visible frame — including the
    # letterbox bars — so every corner handle is visible and draggable in the
    # UI; the user fine-tunes from there.
    h, w = frame_bgr.shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners.tolist()


def detect_tiles(
    frame_bgr: np.ndarray,
    max_tiles: int = MAX_TILES,
    num_white: "int | None" = None,
) -> list[dict[str, Any]]:
    """Convenience: auto-detect the mat corners and project the keys in one call.

    Kept for backward compatibility (/api/autodetect, tests). The reliable path in
    the UI is detect_corners() -> user adjusts -> tiles_from_corners().
    Returns [] when no mat is found.
    """
    corners = detect_corners(frame_bgr, num_white)
    if corners is None:
        return []
    return tiles_from_corners(corners, num_white, max_tiles)
