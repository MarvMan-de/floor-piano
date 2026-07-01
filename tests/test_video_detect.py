"""Tests for webui/video_detect.py — hardware-free, uses synthetic frames."""
import numpy as np
import cv2
import pytest

from webui.video_detect import detect_tile_hits


def _tile(tile_id, polygon, enabled=True):
    return {"id": tile_id, "polygon": polygon, "enabled": enabled}


def _bright_frame(w=640, h=480, brightness=200):
    """Uniformly bright frame."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _dark_frame(w=640, h=480):
    """Uniformly dark frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_bright_region(rx, ry, rw, rh, brightness=220, w=640, h=480):
    frame = _dark_frame(w, h)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (brightness, brightness, brightness), -1)
    return frame


# ── Basic detection ───────────────────────────────────────────────────────

def test_bright_region_triggers_tile():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _frame_with_bright_region(50, 50, 200, 200, brightness=200)
    result = detect_tile_hits(frame, [_tile(1, polygon)], brightness_threshold=80)
    assert 1 in result


def test_dark_region_does_not_trigger():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _dark_frame()
    result = detect_tile_hits(frame, [_tile(1, polygon)], brightness_threshold=80)
    assert 1 not in result


def test_disabled_tile_is_skipped():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _bright_frame()
    result = detect_tile_hits(frame, [_tile(1, polygon, enabled=False)], brightness_threshold=80)
    assert result == []


def test_empty_tiles_returns_empty():
    frame = _bright_frame()
    result = detect_tile_hits(frame, [])
    assert result == []


def test_empty_frame_no_triggers():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _dark_frame()
    result = detect_tile_hits(frame, [_tile(1, polygon)])
    assert result == []


# ── Multiple tiles ────────────────────────────────────────────────────────

def test_only_bright_tile_triggers_among_two():
    poly_bright = [[50, 50], [250, 50], [250, 250], [50, 250]]
    poly_dark   = [[350, 50], [550, 50], [550, 250], [350, 250]]
    # Only the left region is bright
    frame = _frame_with_bright_region(50, 50, 200, 200, brightness=200)
    result = detect_tile_hits(frame, [_tile(1, poly_bright), _tile(2, poly_dark)])
    assert 1 in result
    assert 2 not in result


def test_both_tiles_trigger_when_whole_frame_bright():
    poly_a = [[50, 50], [200, 50], [200, 200], [50, 200]]
    poly_b = [[300, 50], [450, 50], [450, 200], [300, 200]]
    frame = _bright_frame(brightness=200)
    result = detect_tile_hits(frame, [_tile(1, poly_a), _tile(2, poly_b)])
    assert 1 in result
    assert 2 in result


# ── Threshold ─────────────────────────────────────────────────────────────

def test_just_below_threshold_does_not_trigger():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    # Fill polygon region with brightness = 79 (just below threshold of 80)
    frame = _frame_with_bright_region(50, 50, 200, 200, brightness=79)
    result = detect_tile_hits(frame, [_tile(1, polygon)], brightness_threshold=80)
    assert result == []


def test_just_above_threshold_triggers():
    polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _frame_with_bright_region(50, 50, 200, 200, brightness=81)
    result = detect_tile_hits(frame, [_tile(1, polygon)], brightness_threshold=80)
    assert 1 in result


# ── Edge cases ────────────────────────────────────────────────────────────

def test_polygon_outside_frame_does_not_crash():
    # Polygon entirely outside the 640x480 frame
    polygon = [[700, 500], [900, 500], [900, 700], [700, 700]]
    frame = _bright_frame()
    result = detect_tile_hits(frame, [_tile(1, polygon)])
    # Should not raise; result may be empty (no pixels inside frame)
    assert isinstance(result, list)


def test_polygon_with_fewer_than_3_points_is_skipped():
    polygon = [[50, 50], [250, 50]]
    frame = _bright_frame()
    result = detect_tile_hits(frame, [_tile(1, polygon)])
    assert result == []


def test_returns_tile_ids_not_indices():
    poly = [[50, 50], [250, 50], [250, 250], [50, 250]]
    frame = _bright_frame()
    result = detect_tile_hits(frame, [_tile(99, poly)])
    assert 99 in result
