from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from webui.tile_store import NOTE_NAMES, MAX_TILES


def detect_tiles(
    frame_bgr: np.ndarray,
    max_tiles: int = MAX_TILES,
    min_area_px: int = 2000,
) -> list[dict[str, Any]]:
    """Return up to *max_tiles* polygon suggestions from contour detection.

    Pipeline:
      BGR → grayscale → GaussianBlur(5×5) → Canny(50,150)
      → findContours(RETR_EXTERNAL) → filter area + border-touching
      → approxPolyDP(ε = 2% arc length) → sort left→right top→bottom
      → assign note names C4–B5
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suggestions: list[dict[str, Any]] = []
    BORDER = 2  # pixels; contours touching within this margin are skipped

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        # Skip contours that touch the image border
        x, y, cw, ch = cv2.boundingRect(contour)
        if x <= BORDER or y <= BORDER or (x + cw) >= (w - BORDER) or (y + ch) >= (h - BORDER):
            continue

        epsilon = 0.02 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        polygon = approx.reshape(-1, 2).tolist()

        if len(polygon) < 3:
            continue

        suggestions.append({
            "polygon": polygon,
            "_sort_key": (y + ch // 2, x + cw // 2),  # centroid approx for sorting
        })

    # Sort left→right, top→bottom by centroid
    suggestions.sort(key=lambda s: (s["_sort_key"][0] // 50, s["_sort_key"][1]))

    # Trim to max and assign note names
    suggestions = suggestions[:max_tiles]
    for i, s in enumerate(suggestions):
        del s["_sort_key"]
        s["label"] = NOTE_NAMES[i]
        s["note"] = NOTE_NAMES[i]

    return suggestions
