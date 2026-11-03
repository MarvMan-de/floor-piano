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
    # dark trim around the key area (the real mat has one) — it closes each
    # white key into an isolated bright blob, like on the printed vinyl
    cv2.rectangle(mat, (margin - 2, margin - 2),
                  (margin + kw + 2, margin + kh + 2), (25, 25, 25), 3)

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


# ── white-key pattern detection (bright-background robustness) ───────────────

def bright_background_scene():
    """The real-world failure case: a white cupboard/wall band touches the mat.

    A plain brightness mask merges mat + background into one blob, so the old
    biggest-bright-blob quad framed the wrong region. The white-key pattern
    detector must still find the mat.
    """
    scene = synthetic_scene(skew=6)
    scene[:44, :] = 240          # white wall band touching the mat's top edge
    return scene


def test_auto_calibrate_survives_bright_background():
    scene = bright_background_scene()
    corners, info = mc.auto_calibrate(scene, NUM_WHITE)
    assert corners is not None, info
    assert info["method"] == "white_keys"
    warped, _ = mc._warp(scene, corners, CANVAS_W, CANVAS_H)
    dark = mc._dark_mask(warped)
    model = mc.black_key_model(NUM_WHITE, CANVAS_W, CANVAS_H)
    iou = (dark & model).sum() / (dark | model).sum()
    assert iou > 0.55, (iou, info)


def test_quad_from_white_keys_matches_truth():
    scene = synthetic_scene(skew=0)
    quad = mc.quad_from_white_keys(scene, NUM_WHITE)
    assert quad is not None
    # mat KEY AREA at offset 40+margin(14) with size 900x260 (the white-key
    # grid, not the outer mat edge)
    expected = {(54, 54), (54 + 900, 54), (54 + 900, 54 + 260), (54, 54 + 260)}
    for ex, ey in expected:
        assert min(np.hypot(quad[:, 0] - ex, quad[:, 1] - ey)) < 12, quad


def test_quad_from_white_keys_none_without_pattern():
    plain = np.full((300, 400, 3), 110, np.uint8)
    cv2.rectangle(plain, (60, 80), (340, 220), (250, 250, 250), -1)  # plain rug
    assert mc.quad_from_white_keys(plain, NUM_WHITE) is None


def test_white_key_model_points_shape_and_edges():
    pts = mc.white_key_model_points(NUM_WHITE, CANVAS_W, CANVAS_H)
    assert len(pts) == NUM_WHITE
    for (tx, ty), (cx, cy), (bx, by) in pts:
        assert ty == 0.0 and by == float(CANVAS_H)
        assert 0 <= tx <= CANVAS_W and 0 <= bx <= CANVAS_W
        assert 0 < cy < CANVAS_H


def test_orient_corners_survives_rough_user_quad():
    """A user-placed quad is rough: it may cut through the black keys at the
    top and include a strip of floor at the bottom. The mat's bottom shadow
    then forms a full-width dark band that must NOT win the vertical flip
    (regression: all black keys were projected onto the mat's bottom half)."""
    scene = synthetic_scene(skew=0)
    # true key area: (54,54)..(954,314). Rough quad: shifted down so the top
    # edge cuts the blacks and the bottom includes floor.
    rough = np.float32([[60, 130], [950, 130], [950, 400], [60, 400]])
    oriented, _iou = mc.orient_corners(scene, rough, NUM_WHITE)
    # TL/TR must stay the screen-top edge (black keys hang from the top)
    assert oriented[0][1] < oriented[3][1]
    assert oriented[1][1] < oriented[2][1]
    # and C4 on the left (the scene is not mirrored)
    assert oriented[0][0] < oriented[1][0]
