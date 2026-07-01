from __future__ import annotations

"""Tile hit detection on RGB/BGR frames from video playback.

The demo videos record depth-like or natural-light footage. This module uses
mean brightness inside each tile polygon as a simple trigger: if a region is
bright (foot / object present), the tile fires.

This is separate from src/detection.py which works on perspective-warped
16-bit depth arrays. That path requires the Orbbec SDK; this one needs only
NumPy + OpenCV and works on any video file.
"""

from typing import Any

import cv2
import numpy as np

# Default threshold: pixels with mean grayscale above this fire the tile.
DEFAULT_BRIGHTNESS_THRESHOLD = 80


def detect_tile_hits(
    frame_bgr: np.ndarray,
    tiles: list[dict[str, Any]],
    brightness_threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
) -> list[Any]:
    """Return tile ids where mean polygon brightness exceeds *brightness_threshold*.

    Args:
        frame_bgr: Current video frame (H×W×3 uint8).
        tiles: List of tile dicts with "id", "polygon", "enabled" keys.
        brightness_threshold: 0–255; mean grayscale value that counts as a hit.

    Returns:
        List of tile ids (whatever type tile["id"] is) that triggered.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    triggered = []
    for tile in tiles:
        if not tile.get("enabled", True):
            continue
        poly = tile.get("polygon", [])
        if len(poly) < 3:
            continue

        pts = np.array(poly, dtype=np.int32)

        # Clamp to frame bounds before drawing mask
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        pixels = gray[mask > 0]
        if len(pixels) == 0:
            continue

        if float(pixels.mean()) > brightness_threshold:
            triggered.append(tile.get("id"))

    return triggered
