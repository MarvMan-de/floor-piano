"""Tests for webui/depth_detect.py — synthetic depth maps, no hardware.

Detection is a thin "contact band" just above the captured surface: a fingertip
touching fires; a hand hovering/passing well above the surface does NOT.
"""
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


def test_touch_fires_only_that_tile():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150),
                                    _tile(2, 200, 50, 300, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 985  # fingertip 15 mm above the surface (in contact band)
    assert detect_tile_hits_depth(depth, surface, lm, ids) == {1}


def test_hovering_hand_does_not_fire():
    """The reported bug: a hand passing ~6 cm over the keys must NOT trigger."""
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 940  # hand 60 mm in front -> above the contact band
    assert detect_tile_hits_depth(depth, surface, lm, ids) == set()


def test_whole_surface_in_band_does_not_fire_all():
    """The 'all keys fire' bug: a tilted/noisy surface drifting into the contact
    band across the whole frame is one huge blob -> rejected, not every tile."""
    tiles = [_tile(i, (i - 1) * 60 + 10, 50, (i - 1) * 60 + 60, 150) for i in range(1, 6)]
    lm, ids = build_tile_label_map(tiles, W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = np.full((H, W), 985, np.uint16)  # entire frame 15 mm "in front" (drift)
    assert detect_tile_hits_depth(depth, surface, lm, ids) == set()


def test_tilted_surface_contact_fires():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    ramp = np.linspace(800, 1200, W).astype(np.uint16)   # tilted surface
    surface = np.tile(ramp, (H, 1))
    depth = surface.copy()
    depth[70:130, 70:130] = surface[70:130, 70:130] - 15  # 15 mm above local surface
    assert detect_tile_hits_depth(depth, surface, lm, ids) == {1}


def test_surface_noise_does_not_fire():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    surface = np.full((H, W), 1000, np.uint16)
    depth = surface.copy()
    depth[70:130, 70:130] = 998  # only 2 mm -> below contact_min (noise)
    assert detect_tile_hits_depth(depth, surface, lm, ids) == set()


def test_no_surface_returns_empty():
    lm, ids = build_tile_label_map([_tile(1, 50, 50, 150, 150)], W, H)
    assert detect_tile_hits_depth(np.full((H, W), 985, np.uint16), None, lm, ids) == set()


def test_tracker_edge_trigger_and_release():
    t = DepthHitTracker(release_frames=2)
    assert t.update({1}) == {1}      # rising edge -> fire
    assert t.update({1}) == set()    # still held -> no re-fire
    assert t.update(set()) == set()  # miss 1
    assert t.update(set()) == set()  # miss 2 -> released
    assert t.update({1}) == {1}      # can fire again
