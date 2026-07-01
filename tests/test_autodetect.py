"""Tests for webui/autodetect.py — hardware-free, uses synthetic numpy frames."""
import numpy as np
import cv2
import pytest

from webui.autodetect import detect_tiles
from webui.tile_store import NOTE_NAMES


def _blank_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_rect(x, y, w, h, frame_w=640, frame_h=480, rect_color=255):
    """White-filled rectangle on a black background."""
    frame = _blank_frame(frame_w, frame_h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (rect_color, rect_color, rect_color), -1)
    return frame


# ── Basic detection ───────────────────────────────────────────────────────

def test_detects_single_rectangle():
    frame = _frame_with_rect(100, 100, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    assert len(results) >= 1


def test_returns_polygon_with_at_least_3_points():
    frame = _frame_with_rect(100, 100, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    for r in results:
        assert len(r["polygon"]) >= 3


def test_assigns_note_names():
    frame = _frame_with_rect(100, 100, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    for r in results:
        assert r["note"] in NOTE_NAMES
        assert r["label"] in NOTE_NAMES


# ── Filtering ─────────────────────────────────────────────────────────────

def test_filters_small_area():
    # Tiny 10×10 rect — well below the default 2000 px² threshold
    frame = _frame_with_rect(50, 50, 10, 10)
    results = detect_tiles(frame, min_area_px=2000)
    assert len(results) == 0


def test_filters_border_touching_contour():
    # Rectangle starting at x=0 touches the left border
    frame = _frame_with_rect(0, 50, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    assert len(results) == 0


def test_interior_rect_not_filtered():
    # Rectangle well away from the border should survive
    frame = _frame_with_rect(100, 100, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    assert len(results) >= 1


# ── Sorting & note assignment ─────────────────────────────────────────────

def test_assigns_note_names_left_to_right():
    """Two rects side by side: left one should get the lower note."""
    frame = _blank_frame()
    # Left rect
    cv2.rectangle(frame, (50, 100), (200, 300), (255, 255, 255), -1)
    # Right rect
    cv2.rectangle(frame, (300, 100), (500, 300), (255, 255, 255), -1)
    results = detect_tiles(frame, min_area_px=1000)
    if len(results) >= 2:
        left_note_idx  = NOTE_NAMES.index(results[0]["note"])
        right_note_idx = NOTE_NAMES.index(results[1]["note"])
        assert left_note_idx < right_note_idx


def test_caps_at_24_tiles():
    """Even if we have many contours, never return more than 24."""
    frame = _blank_frame(1280, 960)
    # Draw a 5x5 grid of small rects, each large enough to pass area filter
    for row in range(5):
        for col in range(5):
            x = 100 + col * 220
            y = 100 + row * 170
            cv2.rectangle(frame, (x, y), (x + 80, y + 60), (255, 255, 255), -1)
    results = detect_tiles(frame, max_tiles=24, min_area_px=500)
    assert len(results) <= 24


def test_empty_frame_returns_no_tiles():
    frame = _blank_frame()
    results = detect_tiles(frame)
    assert results == []


def test_result_polygon_points_are_integers():
    frame = _frame_with_rect(100, 100, 200, 150)
    results = detect_tiles(frame, min_area_px=1000)
    for r in results:
        for pt in r["polygon"]:
            assert isinstance(pt[0], (int, np.integer))
            assert isinstance(pt[1], (int, np.integer))
