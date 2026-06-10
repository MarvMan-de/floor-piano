"""Tests for the geometry, frame-decoding and calibration helpers in detection.py.

All hardware-free: synthetic numpy arrays only.
"""
import numpy as np
import pytest

import constants
import detection as d


# --- key_bounds ------------------------------------------------------------

def test_key_bounds_divisible_tiles_exactly():
    bounds = d.key_bounds(700, 7)
    assert bounds == [0, 100, 200, 300, 400, 500, 600, 700]


def test_key_bounds_non_divisible_covers_full_width_without_gap():
    width, n = 701, 7
    bounds = d.key_bounds(width, n)
    assert len(bounds) == n + 1
    assert bounds[0] == 0
    assert bounds[-1] == width            # no silent gap at the right edge
    assert bounds == sorted(bounds)       # monotonic
    # every column has positive width
    assert all(bounds[i + 1] > bounds[i] for i in range(n))


def test_key_bounds_rejects_non_positive():
    with pytest.raises(ValueError):
        d.key_bounds(700, 0)


def test_detect_hits_non_divisible_width_detects_last_key():
    width, n = 701, 7
    frame = d.flat_floor_frame(50, width, 1000)
    frame = d.stamp_foot(frame, key_index=n - 1, num_keys=n, foot_depth=700)
    assert d.detect_hits(frame, n, 1000, 50) == {n - 1}


# --- decode_depth ----------------------------------------------------------

def test_decode_depth_correct_size():
    h, w = 4, 5
    raw = np.arange(h * w, dtype=np.uint16).tobytes()
    out = d.decode_depth(raw, h, w)
    assert out.shape == (h, w)
    assert int(out[0, 0]) == 0
    assert int(out[-1, -1]) == h * w - 1


def test_decode_depth_wrong_size_returns_none():
    # 3 bytes can never be a 4x5 uint16 frame -> guard returns None, no crash.
    assert d.decode_depth(b"\x00\x00\x00", 4, 5) is None


# --- scale_point_to_depth --------------------------------------------------

def test_scale_point_to_depth_scales_by_resolution_ratio():
    # RGB 640x480 -> depth 320x240 halves the coordinates.
    x, y = d.scale_point_to_depth((100, 200), rgb_shape=(480, 640, 3), depth_shape=(240, 320))
    assert (round(x), round(y)) == (50, 100)


# --- sample_floor_depth ----------------------------------------------------

def test_sample_floor_depth_returns_median_in_bounds():
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    assert d.sample_floor_depth(depth, (50, 50), (100, 100, 3)) == 1000


def test_sample_floor_depth_near_edge_returns_none():
    # No silent default: a wrong floor depth makes the piano dead or constantly
    # firing, so "couldn't measure" must be explicit (calibrate.py retries).
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    assert d.sample_floor_depth(depth, (0, 0), (100, 100, 3)) is None


def test_sample_floor_depth_all_invalid_returns_none():
    depth = np.zeros((100, 100), dtype=np.uint16)
    assert d.sample_floor_depth(depth, (50, 50), (100, 100, 3)) is None


# --- build_config ----------------------------------------------------------

def _corner_points():
    return {
        0: np.array([0.0, 0.0]),
        1: np.array([10.0, 0.0]),
        2: np.array([10.0, 10.0]),
        3: np.array([0.0, 10.0]),
    }


def test_build_config_orders_corners_and_is_valid():
    cfg = d.build_config(_corner_points(), floor_depth=950, rgb_shape=(480, 640, 3))
    assert cfg["corners"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert cfg["num_white_keys"] == constants.DEFAULT_NUM_WHITE
    assert cfg["canvas_size"] == [640, 480]
    assert cfg["floor_depth"] == 950
    assert d.validate_config(cfg) is True


def test_build_config_respects_custom_num_white():
    cfg = d.build_config(_corner_points(), 950, (480, 640, 3), num_white=7)
    assert cfg["num_white_keys"] == 7
