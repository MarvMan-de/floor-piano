"""Tests for webui/tile_store.py — no camera, no FastAPI needed."""
import json
import os
import pathlib
import pytest

from webui.tile_store import (
    NOTE_NAMES,
    default_tiles_doc,
    load_tiles,
    save_tiles,
    validate_tiles_doc,
)


def _tile(note="C4", polygon=None):
    return {
        "id": 1,
        "label": note,
        "note": note,
        "polygon": polygon or [[0, 0], [100, 0], [100, 100], [0, 100]],
        "color": "#4A90D9",
        "enabled": True,
    }


# ── default_tiles_doc ─────────────────────────────────────────────────────

def test_default_doc_has_empty_tiles():
    doc = default_tiles_doc()
    assert doc["tiles"] == []


def test_default_doc_has_frame_size():
    doc = default_tiles_doc()
    assert doc["frame_width"] == 640
    assert doc["frame_height"] == 480


# ── validate_tiles_doc ───────────────────────────────────────────────────

def test_validate_accepts_valid_doc():
    doc = default_tiles_doc()
    doc["tiles"] = [_tile()]
    validate_tiles_doc(doc)  # should not raise


def test_validate_rejects_more_than_24_tiles():
    doc = default_tiles_doc()
    doc["tiles"] = [_tile(NOTE_NAMES[i % len(NOTE_NAMES)]) for i in range(25)]
    with pytest.raises(ValueError, match="Too many tiles"):
        validate_tiles_doc(doc)


def test_validate_rejects_invalid_note_name():
    doc = default_tiles_doc()
    doc["tiles"] = [_tile(note="X9")]
    with pytest.raises(ValueError, match="invalid note name"):
        validate_tiles_doc(doc)


def test_validate_rejects_polygon_under_3_points():
    doc = default_tiles_doc()
    doc["tiles"] = [_tile(polygon=[[0, 0], [100, 0]])]
    with pytest.raises(ValueError, match="at least 3 points"):
        validate_tiles_doc(doc)


def test_validate_rejects_non_list_tiles():
    with pytest.raises(ValueError, match="must be a list"):
        validate_tiles_doc({"tiles": "not a list"})


def test_validate_all_24_notes_accepted():
    doc = default_tiles_doc()
    doc["tiles"] = [_tile(note=n) for n in NOTE_NAMES]
    validate_tiles_doc(doc)  # should not raise


# ── load_tiles ───────────────────────────────────────────────────────────

def test_load_returns_default_when_file_absent(tmp_path):
    missing = tmp_path / "nonexistent.json"
    result = load_tiles(missing)
    assert result == default_tiles_doc()


def test_load_reads_existing_file(tmp_path):
    p = tmp_path / "tiles.json"
    doc = default_tiles_doc()
    doc["tiles"] = [_tile()]
    p.write_text(json.dumps(doc))
    result = load_tiles(p)
    assert len(result["tiles"]) == 1


# ── save_tiles ───────────────────────────────────────────────────────────

def test_save_writes_valid_json(tmp_path):
    p = tmp_path / "tiles.json"
    doc = default_tiles_doc()
    doc["tiles"] = [_tile()]
    save_tiles(doc, p)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert len(loaded["tiles"]) == 1


def test_atomic_save_no_tmp_left_behind(tmp_path):
    p = tmp_path / "tiles.json"
    doc = default_tiles_doc()
    save_tiles(doc, p)
    tmp = p.with_suffix(".tmp")
    assert not tmp.exists()


def test_save_read_round_trip(tmp_path):
    p = tmp_path / "tiles.json"
    doc = default_tiles_doc()
    doc["tiles"] = [_tile("G4", [[10, 20], [30, 20], [30, 40], [10, 40]])]
    save_tiles(doc, p)
    loaded = load_tiles(p)
    assert loaded["tiles"][0]["note"] == "G4"
    assert loaded["tiles"][0]["polygon"] == [[10, 20], [30, 20], [30, 40], [10, 40]]


def test_save_rejects_invalid_doc(tmp_path):
    p = tmp_path / "tiles.json"
    doc = default_tiles_doc()
    doc["tiles"] = [_tile(note="INVALID")]
    with pytest.raises(ValueError):
        save_tiles(doc, p)
    assert not p.exists()  # file must NOT be created on validation failure


def test_save_creates_parent_dir(tmp_path):
    p = tmp_path / "nested" / "dir" / "tiles.json"
    doc = default_tiles_doc()
    save_tiles(doc, p)
    assert p.exists()
