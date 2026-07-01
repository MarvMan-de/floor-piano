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
        assert len(t["polygon"]) == 4
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
    p = tiles[0]["polygon"]  # C4, leftmost white key
    top = ((p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2) ** 0.5
    bottom = ((p[3][0] - p[2][0]) ** 2 + (p[3][1] - p[2][1]) ** 2) ** 0.5
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
