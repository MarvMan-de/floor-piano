"""Tests for webui/depth_detect.py — synthetic depth maps, no hardware."""
import numpy as np

from webui.depth_detect import (DepthHitTracker, build_tile_label_map,
                               detect_tile_hits_depth)

W, H = 640, 480


def _tile(tid, x0, y0, x1, y1):
    return {"id": tid, "enabled": True,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_label_map_indices():
    lm, ids = build_tile_label_map([_tile(10, 0, 0, 100, 100),
                                    _tile(20, 100, 0, 200, 100)], W, H)
    assert ids == [10, 20]
    assert lm[50, 50] == 0 and lm[50, 150] == 1 and lm[300, 300] == -1


def test_press_in_one_tile_fires_only_it():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150),
                                    _tile(2, 200, 50, 300, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 950  # finger 50 mm in front of the surface, inside tile 1
    assert detect_tile_hits_depth(depth, surface, lm, ids) == {1}


def test_tilted_surface_background_subtraction():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    ramp = np.linspace(800, 1200, W).astype(np.uint16)   # depth varies left->right
    surface = np.tile(ramp, (H, 1))
    depth = surface.copy()
    depth[70:130, 70:130] = surface[70:130, 70:130] - 60  # 60 mm above the local surface
    assert detect_tile_hits_depth(depth, surface, lm, ids) == {1}


def test_noise_under_threshold_does_not_fire():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 990  # only 10 mm closer, below the 20 mm threshold
    assert detect_tile_hits_depth(depth, surface, lm, ids, threshold_mm=20) == set()


def test_far_above_surface_ignored():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 500  # 500 mm above surface -> arm/hand, beyond max_height
    assert detect_tile_hits_depth(depth, surface, lm, ids, max_height_mm=200) == set()


def test_no_surface_returns_empty():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    assert detect_tile_hits_depth(np.full((H, W), 950, np.uint16), None, lm, ids) == set()


def test_tracker_edge_trigger_and_release():
    t = DepthHitTracker(release_frames=2)
    assert t.update({1}) == {1}      # rising edge -> fire
    assert t.update({1}) == set()    # still held -> no re-fire
    assert t.update(set()) == set()  # miss 1
    assert t.update(set()) == set()  # miss 2 -> released
    assert t.update({1}) == {1}      # can fire again
