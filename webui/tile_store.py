from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent
TILES_PATH = _REPO_ROOT / "config" / "tiles.json"

NOTE_NAMES: list[str] = [
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
]
MAX_TILES = 24


def default_tiles_doc() -> dict[str, Any]:
    return {
        "tiles": [],
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
    }


def load_tiles(path: Path = TILES_PATH) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default_tiles_doc()


def save_tiles(doc: dict[str, Any], path: Path = TILES_PATH) -> None:
    """Atomic write: serialize to a sibling .tmp file, then os.replace()."""
    validate_tiles_doc(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def validate_tiles_doc(doc: dict[str, Any]) -> None:
    """Raise ValueError on schema violations."""
    tiles = doc.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("'tiles' must be a list")
    if len(tiles) > MAX_TILES:
        raise ValueError(f"Too many tiles: {len(tiles)} > {MAX_TILES}")
    for i, tile in enumerate(tiles):
        note = tile.get("note", "")
        if note not in NOTE_NAMES:
            raise ValueError(f"Tile {i}: invalid note name '{note}'")
        poly = tile.get("polygon", [])
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError(f"Tile {i}: polygon must have at least 3 points")
        for pt in poly:
            if not (isinstance(pt, list) and len(pt) == 2):
                raise ValueError(f"Tile {i}: each polygon point must be [x, y]")
