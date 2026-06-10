"""Mat-based auto-calibration on a synthetic piano mat with known geometry."""
import cv2
import numpy as np
import pytest

import constants
import detection as d
import mat_calibration as mc

NUM_WHITE = 14
CANVAS_W, CANVAS_H = constants.TARGET_WIDTH, constants.TARGET_HEIGHT


def synthetic_scene(rotate=0, mirror=False, skew=8):
    """A printed piano mat (white, black keys + separator lines) on a darker floor.

    Geometry mirrors the real test videos: mat almost fills the frame, slight
    perspective skew, a thin white margin between the painted keys and the mat
    edge. Returns the BGR image.
    """
    # key area drawn flat first
    kw, kh = 900, 260
    margin = 14
    mat = np.full((kh + 2 * margin, kw + 2 * margin, 3), 245, np.uint8)
    keys = d.build_keyboard(NUM_WHITE, kw, kh)
    for k in keys:
        if k.kind == "black":
            cv2.rectangle(mat, (margin + k.x0, margin + k.y0),
                          (margin + k.x1, margin + k.y1), (25, 25, 25), -1)
    for x in d.key_bounds(kw, NUM_WHITE)[1:-1]:
        cv2.line(mat, (margin + x, margin), (margin + x, margin + kh), (25, 25, 25), 3)

    # place on a darker floor with a mild perspective skew
    floor = np.full((kh + 2 * margin + 80, kw + 2 * margin + 80, 3), 110, np.uint8)
    h, w = mat.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40 + skew, 40], [40 + w - skew / 2, 40 + skew / 2],
                      [40 + w, 40 + h - skew], [40, 40 + h]])
    M = cv2.getPerspectiveTransform(src, dst)
    scene = cv2.warpPerspective(mat, M, (floor.shape[1], floor.shape[0]),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(110, 110, 110))
    if mirror:
        scene = scene[:, ::-1]
    for _ in range(rotate // 90):
        scene = cv2.rotate(scene, cv2.ROTATE_90_CLOCKWISE)
    return scene


def calibrated_iou(scene):
    corners, info = mc.auto_calibrate(scene, NUM_WHITE)
    assert corners is not None, info
    warped, _ = mc._warp(scene, corners, CANVAS_W, CANVAS_H)
    dark = mc._dark_mask(warped)
    model = mc.black_key_model(NUM_WHITE, CANVAS_W, CANVAS_H)
    return (dark & model).sum() / (dark | model).sum(), info


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_auto_calibrate_resolves_every_rotation(rotate):
    iou, info = calibrated_iou(synthetic_scene(rotate=rotate))
    assert iou > 0.55, info


def test_auto_calibrate_resolves_mirrored_source():
    """Front-camera clips are mirrored — a flip the warp must fold in."""
    iou, info = calibrated_iou(synthetic_scene(mirror=True))
    assert iou > 0.55, info


def test_auto_calibrate_reports_refinement():
    corners, info = mc.auto_calibrate(synthetic_scene(), NUM_WHITE)
    assert info["refined"] is True
    assert info["matched_bars"] >= 8
    assert info["residual_after"] < 6.0


def test_no_mat_returns_none():
    floor_only = np.full((300, 400, 3), 110, np.uint8)
    corners, info = mc.auto_calibrate(floor_only, NUM_WHITE)
    assert corners is None
    assert "reason" in info


def test_find_mat_quad_corners_near_truth():
    scene = synthetic_scene(skew=0)
    quad = mc.find_mat_quad(scene)
    assert quad is not None
    # mat placed at offset 40 with size 928x288
    expected = {(40, 40), (40 + 928, 40), (40 + 928, 40 + 288), (40, 40 + 288)}
    for ex, ey in expected:
        assert min(np.hypot(quad[:, 0] - ex, quad[:, 1] - ey)) < 8


def test_median_background_removes_moving_foot():
    frames = []
    for i in range(9):
        f = synthetic_scene()
        cv2.circle(f, (100 + 60 * i, 150), 25, (60, 40, 30), -1)  # a moving shoe
        frames.append(f)
    bg = mc.median_background(frames)
    clean = synthetic_scene()
    assert np.abs(bg.astype(int) - clean.astype(int)).max() < 30
