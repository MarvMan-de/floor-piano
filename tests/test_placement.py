"""Tests for the placement coach (pure geometry — no camera needed)."""
import numpy as np
import pytest

from placement import assess_placement, status_to_dict

SHAPE = (480, 640, 3)  # h, w, c


def quad(x0, y0, x1, y1):
    """A TL,TR,BR,BL corner array for an axis-aligned box."""
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_no_mat():
    s = assess_placement(None, SHAPE)
    assert s.code == "no_mat" and s.ok is False


def test_ok_centered_with_margin():
    s = assess_placement(quad(120, 120, 520, 360), SHAPE)
    assert s.code == "ok" and s.ok is True


def test_no_depth_overrides_geometry():
    # Even a perfectly placed mat is unusable without depth at its centre.
    s = assess_placement(quad(120, 120, 520, 360), SHAPE, depth_ok=False)
    assert s.code == "no_depth" and s.ok is False


def test_clipped_left_points_camera_left():
    s = assess_placement(quad(-30, 150, 400, 330), SHAPE)
    assert s.code == "clipped"
    assert "left" in s.metrics["clipped"]
    assert "left" in s.hint.lower()


def test_clipped_opposite_edges_means_too_big():
    # Mat wider than the frame on both sides -> "doesn't fit, more distance".
    s = assess_placement(quad(-40, 150, 680, 330), SHAPE)
    assert s.code == "clipped"
    assert "does not fit" in s.hint.lower()


def test_near_edge_top_is_tight_not_clipped():
    # Inside the frame (y0=5) but within the 4% margin band -> near_edge.
    s = assess_placement(quad(120, 5, 520, 360), SHAPE)
    assert s.code == "near_edge"
    assert "top" in s.metrics["tight"]


def test_too_small_is_usable():
    s = assess_placement(quad(300, 220, 360, 260), SHAPE)
    assert s.code == "too_small" and s.ok is True


@pytest.mark.parametrize("corners", [None, quad(120, 120, 520, 360)])
def test_status_to_dict_is_json_friendly(corners):
    import json
    d = status_to_dict(assess_placement(corners, SHAPE))
    json.dumps(d)  # must not raise
    assert set(d) == {"ok", "code", "headline", "hint", "metrics"}
