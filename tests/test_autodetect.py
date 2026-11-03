"""Tests for webui/autodetect.py — hardware-free, synthetic numpy frames.

The autodetect was reworked to reuse the project's mat calibration + keyboard
geometry: the 4 mat corners drive a perspective projection of the known 24-key
grid. These tests cover that projection (the reliable path) and the best-effort
corner auto-detection.
"""
import numpy as np
import pytest

from webui.autodetect import (_content_roi, detect_corners, detect_tiles,
                              tiles_from_corners)
from webui.tile_store import NOTE_NAMES


def _blank(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── tiles_from_corners: the reliable path ──────────────────────────────────

def test_projects_full_24_key_grid():
    tiles = tiles_from_corners([[100, 60], [560, 60], [560, 300], [100, 300]])
    assert len(tiles) == 24
    for t in tiles:
        assert t["note"] in NOTE_NAMES
        assert t["label"] in NOTE_NAMES
        assert 4 <= len(t["polygon"]) <= 8   # rectangles or notched piano shapes
        for x, y in t["polygon"]:
            assert isinstance(x, int) and isinstance(y, int)


def test_axis_aligned_corners_give_axis_aligned_keys():
    tiles = tiles_from_corners([[0, 0], [640, 0], [640, 480], [0, 480]])
    # First white key hugs the left edge, full height.
    poly = tiles[0]["polygon"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == 0 and min(ys) == 0 and max(ys) == 480


def test_tilted_corners_give_perspective_trapezoids():
    # Keystone: top edge narrower than bottom -> keys must be trapezoids.
    tiles = tiles_from_corners([[200, 80], [440, 80], [560, 300], [80, 300]])
    p = tiles[0]["polygon"]  # C4, leftmost white key (notched: 6 points)
    top = ((p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2) ** 0.5
    bottom = ((p[-1][0] - p[-2][0]) ** 2 + (p[-1][1] - p[-2][1]) ** 2) ** 0.5
    assert bottom > top * 1.3  # clearly foreshortened at the top


def test_has_white_and_black_keys():
    tiles = tiles_from_corners([[100, 60], [560, 60], [560, 300], [100, 300]])
    notes = [t["note"] for t in tiles]
    assert "C4" in notes and "C#4" in notes and "B5" in notes
    assert sum("#" in n for n in notes) == 10  # 10 black keys


@pytest.mark.parametrize("bad", [[[0, 0]], [[0, 0], [1, 1], [2, 2]], []])
def test_rejects_wrong_corner_count(bad):
    with pytest.raises((ValueError, TypeError)):
        tiles_from_corners(bad)


# ── auto-detection: graceful on a mat-less frame ───────────────────────────

def test_detect_corners_none_on_blank():
    assert detect_corners(_blank()) is None


def test_detect_tiles_empty_on_blank():
    assert detect_tiles(_blank()) == []


# ── letterbox handling ─────────────────────────────────────────────────────

def test_content_roi_strips_black_bars():
    frame = _blank()
    frame[60:420, :] = 200  # content band with 60px black bars top/bottom
    content, xo, yo = _content_roi(frame)
    assert yo == 60
    assert content.shape[0] == 360


# ── order_corners: any click order -> TL,TR,BR,BL ──────────────────────────

def test_order_corners_all_permutations_agree():
    import itertools
    from webui.autodetect import order_corners
    quad = [[100, 60], [560, 80], [590, 300], [80, 280]]
    expected, _ = order_corners(quad)
    for perm in itertools.permutations(quad):
        got, _how = order_corners(list(perm))
        assert np.allclose(got, expected), (perm, got)


def test_order_corners_oblique_quad_regression():
    """The live-UI failure: an oblique mat quad where the old client-side
    sum/diff sort picked the SAME point as both TL and BL, collapsing the
    perspective transform (all 24 keys became one line at the frame edge)."""
    from webui.autodetect import order_corners
    clicks = [[10, 287], [104, 201], [547, 215], [635, 315]]
    ordered, _how = order_corners(clicks)
    # 4 distinct corners, sane area, and every input point survives
    assert len({tuple(p) for p in ordered.tolist()}) == 4
    assert {tuple(map(round, p)) for p in ordered.tolist()} == \
        {tuple(p) for p in clicks}
    import cv2
    assert cv2.contourArea(ordered) > 40000
    tiles = tiles_from_corners(ordered)
    xs = [p[0] for t in tiles for p in t["polygon"]]
    assert max(xs) - min(xs) > 500  # keys span the mat, not collapsed


def test_order_corners_geometric_fallback_orientation():
    from webui.autodetect import order_corners
    ordered, how = order_corners([[560, 300], [100, 60], [560, 60], [100, 300]])
    assert how == "geometry"
    assert np.allclose(ordered, [[100, 60], [560, 60], [560, 300], [100, 300]])


@pytest.mark.parametrize("bad", [
    [[0, 0], [0, 0], [100, 0], [100, 100]],        # coincident points
    [[0, 0], [50, 0], [100, 0], [150, 0]],         # collinear
    [[0, 0], [1, 0], [1, 1], [0, 1]],              # far too small
])
def test_order_corners_rejects_degenerate(bad):
    from webui.autodetect import order_corners
    with pytest.raises(ValueError):
        order_corners(bad)


def test_order_corners_orients_against_frame():
    """With a frame, the black-key pattern must win over screen geometry."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_mat_calibration import synthetic_scene, NUM_WHITE
    import mat_calibration as mc
    from webui.autodetect import order_corners
    scene = synthetic_scene(skew=0)
    truth, info = mc.auto_calibrate(scene, NUM_WHITE)
    assert truth is not None
    shuffled = [truth[2].tolist(), truth[0].tolist(),
                truth[3].tolist(), truth[1].tolist()]
    ordered, how = order_corners(shuffled, scene)
    assert how == "pattern"
    # same orientation as the calibrated truth: corner-wise distance small
    assert np.abs(np.float32(ordered) - np.float32(truth)).max() < 8, \
        (ordered, truth)

def test_order_corners_keys_down_mat_falls_back_to_geometry():
    """Mat lying KEYS-DOWN: the camera sees only the blank back, so there is
    no pattern to orient by — geometric fallback plus the user's flips."""
    from webui.autodetect import flip_corners, order_corners
    frame = np.full((480, 640, 3), 120, np.uint8)          # wood floor
    frame[180:300, 60:580] = 235                           # blank mat back
    ordered, how = order_corners([[60, 180], [580, 180], [580, 300], [60, 300]],
                                 frame)
    assert how == "geometry"
    assert np.allclose(ordered, [[60, 180], [580, 180], [580, 300], [60, 300]])


def test_flip_corners_h_and_v():
    from webui.autodetect import flip_corners
    tl, tr, br, bl = [0, 0], [100, 0], [100, 40], [0, 40]
    assert np.allclose(flip_corners([tl, tr, br, bl], flip_h=True),
                       [tr, tl, bl, br])
    assert np.allclose(flip_corners([tl, tr, br, bl], flip_v=True),
                       [bl, br, tr, tl])
    # both = 180 degrees; C4 tile must land on the far end after flip_h
    tiles = tiles_from_corners(flip_corners([tl, tr, br, bl], flip_h=True))
    c4_xs = [p[0] for p in tiles[0]["polygon"]]
    assert min(c4_xs) >= 90  # C4 now at the right end of the quad

def test_mat_rotated_180_black_keys_at_bottom():
    """Mat physically rotated 180 deg: black keys along the BOTTOM edge of the
    camera image. The pattern orientation must map the canvas top (where the
    model's black keys live) onto that bottom edge — i.e. TL/TR end up BELOW
    BL/BR on screen, and the projected black-key tiles sit at the bottom."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_mat_calibration import synthetic_scene
    from webui.autodetect import order_corners
    scene = synthetic_scene(rotate=180)
    h, w = scene.shape[:2]
    quad = [[40, 40], [w - 40, 40], [w - 40, h - 40], [40, h - 40]]
    ordered, how = order_corners(quad, scene)
    assert how == "pattern"
    tl, tr, br, bl = ordered
    assert tl[1] > bl[1] and tr[1] > br[1]   # canvas-top edge = screen bottom
    tiles = tiles_from_corners(ordered)
    blacks = [t for t in tiles if "#" in t["note"]]
    whites = [t for t in tiles if "#" not in t["note"]]
    black_y = np.mean([p[1] for t in blacks for p in t["polygon"]])
    white_y = np.mean([p[1] for t in whites for p in t["polygon"]])
    assert black_y > white_y   # black keys projected onto the lower mat half

def test_white_and_black_tiles_do_not_overlap():
    """Pressing F#4 must not also cover the neighbouring whites: white keys
    get the notched piano shape, so black rectangles claim their area alone."""
    import cv2
    tiles = tiles_from_corners([[0, 0], [640, 0], [640, 480], [0, 480]])
    whites = np.zeros((480, 640), np.uint8)
    blacks = np.zeros((480, 640), np.uint8)
    for t in tiles:
        target = blacks if "#" in t["note"] else whites
        cv2.fillPoly(target, [np.asarray(t["polygon"], np.int32)], 1)
    # shared polygon EDGES rasterize into both masks — erode one side so only
    # true interior overlap counts
    blacks_core = cv2.erode(blacks, np.ones((3, 3), np.uint8))
    inter = int((whites & blacks_core).sum())
    assert inter == 0, inter
