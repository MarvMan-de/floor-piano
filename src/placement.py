"""Placement coach: turn 'hunt for the exact spot' into 'follow the hints'.

Given the mat corners a calibration frame detected (or None) and the frame size,
``assess_placement`` returns a STRUCTURED verdict — is the mat fully in view with
margin? if not, which way should the camera move? — instead of a yes/no.

The output is deliberately data, not a printed string, so the same verdict drives
the calibration CLI *and* the planned Pi-hotspot web UI (built separately). The
``code`` field is the stable contract (localise / restyle in the UI as you like);
``headline``/``hint`` are ready-made English text for logs and a quick CLI.

Pure and hardware-free: every branch is unit-tested with synthetic corners, so the
ease-of-use logic is verified long before the camera is attached.

Status codes
------------
    no_mat     — nothing detected; aim the camera at the mat
    no_depth   — mat found but no valid depth at its centre (out of depth view)
    clipped    — mat runs off one or more frame edges
    near_edge  — mat fully inside but hugging an edge (no safety margin)
    too_small  — mat fits with margin but fills little of the frame (low res); OK
    ok         — fully in view with healthy margin
"""

from collections import namedtuple

import numpy as np

# ok:        is this good enough to calibrate from? (too_small / ok are True)
# code:      stable machine key (see module docstring) — the web-UI contract
# headline:  one-line human summary
# hint:      what to do next
# metrics:   numbers behind the verdict (margins, fill, offending edges)
PlacementStatus = namedtuple("PlacementStatus", "ok code headline hint metrics")


def status_to_dict(status):
    """Plain dict (JSON-serialisable) for the web UI / a status file."""
    d = status._asdict()
    d["metrics"] = dict(status.metrics)
    return d


def _move_hint(edges):
    """Translate offending frame edges into a camera move instruction."""
    edges = set(edges)
    # Opposite edges both bad, or three+ edges: the mat simply doesn't fit —
    # the camera needs to see a larger area, i.e. more distance / height.
    if len(edges) >= 3 or {"left", "right"} <= edges or {"top", "bottom"} <= edges:
        return "Mat does not fit — raise the camera or move it further from the mat."
    move = {"left": "left", "right": "right", "top": "up", "bottom": "down"}
    where = " and ".join(move[e] for e in ("top", "bottom", "left", "right") if e in edges)
    return f"Mat is cut off on the {where} — pan/shift the camera {where}."


def assess_placement(corners, frame_shape, depth_ok=True,
                     margin_frac=0.04, fill_frac_min=0.18):
    """Assess one calibration frame's mat placement.

    Parameters
    ----------
    corners : array of 4 (x, y) points in the detection frame, or None if the
        mat/markers were not found this frame.
    frame_shape : the detection frame's shape (h, w[, c]).
    depth_ok : False when the mat was found but no valid depth exists at its
        centre (mat outside the depth FOV, or too close/far).
    margin_frac : a corner closer than this fraction of min(w, h) to an edge
        counts as "tight" (no safety margin) even though it is still inside.
    fill_frac_min : below this fraction of the frame area the mat is "small"
        (usable but low resolution).
    """
    h, w = frame_shape[:2]
    if corners is None:
        return PlacementStatus(
            False, "no_mat", "Mat not detected",
            "Point the camera roughly at the mat and keep it clear of feet.", {})

    pts = np.asarray(corners, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    # Distance from the mat's bounding box to each frame edge; negative = the
    # mat extends past that edge (clipped).
    margins = {"left": x0, "right": w - x1, "top": y0, "bottom": h - y1}
    band = margin_frac * min(w, h)
    clipped = [e for e in ("top", "bottom", "left", "right") if margins[e] < 0]
    tight = [e for e in ("top", "bottom", "left", "right") if 0 <= margins[e] < band]
    fill = (max(x1 - x0, 0.0) * max(y1 - y0, 0.0)) / float(w * h)
    metrics = {
        "margins_px": {k: round(v, 1) for k, v in margins.items()},
        "fill_frac": round(fill, 3),
        "clipped": clipped,
        "tight": tight,
    }

    if not depth_ok:
        return PlacementStatus(
            False, "no_depth", "Mat found but no depth at its centre",
            "The mat is outside the depth view or too close/far — re-aim or adjust height.",
            metrics)
    if clipped:
        return PlacementStatus(
            False, "clipped", f"Mat runs off the frame ({', '.join(clipped)})",
            _move_hint(clipped), metrics)
    if tight:
        return PlacementStatus(
            False, "near_edge", f"Mat is hugging the edge ({', '.join(tight)})",
            _move_hint(tight), metrics)
    if fill < fill_frac_min:
        return PlacementStatus(
            True, "too_small", "Mat fits but is small in the frame",
            "Good enough to calibrate; move the camera closer/lower for more resolution.",
            metrics)
    return PlacementStatus(
        True, "ok", "Good placement — mat fully in view with margin",
        "Hold still while it calibrates.", metrics)
