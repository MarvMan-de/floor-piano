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

    tiles: list[dict[str, Any]] = []
    for key in build_keyboard(num_white, w, h)[:max_tiles]:
        rect = np.float32([[[key.x0, key.y0], [key.x1, key.y0],
                            [key.x1, key.y1], [key.x0, key.y1]]])
        pts = cv2.perspectiveTransform(rect, to_image)[0]
        polygon = [[int(round(px)), int(round(py))] for px, py in pts]
        tiles.append({"polygon": polygon, "label": key.name, "note": key.name})
    return tiles


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
