"""Tests for webui/depth_detect.py — synthetic depth maps, no hardware.

The PressDetector must satisfy: nothing moves -> nothing fires (motion gate);
static artifacts near the surface never fire (artifact mask); a real press
(appears + sits in the contact band) fires exactly once until released.
"""
import numpy as np

from webui.depth_detect import (DepthHitTracker, PressDetector,
                               build_tile_label_map)

W, H = 640, 480


def _tile(tid, x0, y0, x1, y1):
    return {"id": tid, "enabled": True,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _detector(tiles, surface):
    det = PressDetector(surface)
    det.set_tiles(*build_tile_label_map(tiles, W, H))
    return det


def _flat(v=1000):
    return np.full((H, W), v, np.uint16)


def _press(surface, x0, y0, x1, y1, mm=20):
    f = surface.copy()
    f[y0:y1, x0:x1] = surface[y0:y1, x0:x1] - mm
    return f


def test_label_map_indices():
    lm, ids = build_tile_label_map([_tile(10, 0, 0, 100, 100),
                                    _tile(20, 100, 0, 200, 100)], W, H)
    assert ids == [10, 20]
    assert lm[50, 50] == 0 and lm[50, 150] == 1 and lm[300, 300] == -1


def test_static_in_band_region_never_fires():
    """THE bug: tiles kept firing with nothing moving. A region that sits in the
    contact band from the start has no motion -> must never fire."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    frame = _press(surface, 60, 60, 140, 140)  # in-band, but constant forever
    fired = set()
    for _ in range(30):
        fired |= det.update(frame)
    assert fired == set()


def test_real_press_fires_once():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    empty, press = surface.copy(), _press(surface, 70, 70, 130, 130)
    det.update(empty)
    det.update(empty)
    fired = set()
    for _ in range(6):          # finger arrives and stays
        fired |= det.update(press)
    assert fired == {1}
    for _ in range(10):         # held -> no machine-gunning
        assert det.update(press) == set()


def test_press_fires_correct_tile_only():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150), _tile(2, 200, 50, 300, 150)], surface)
    press = _press(surface, 210, 60, 290, 140)
    det.update(surface)
    fired = set()
    for _ in range(4):
        fired |= det.update(press)
    assert fired == {2}


def test_hovering_hand_does_not_fire():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    hover = _press(surface, 60, 60, 140, 140, mm=80)  # 80mm above -> outside band
    det.update(surface)
    fired = set()
    for _ in range(6):
        fired |= det.update(hover)
    assert fired == set()


def test_calibrated_artifacts_never_fire():
    """Pixels already in-band during the empty calibration are masked out, even
    if depth noise makes them flicker (i.e. produce motion) later."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    artifact = _press(surface, 60, 60, 140, 140, mm=15)   # bogus in-band region
    det.calibrate_artifacts([artifact, artifact])
    det.update(surface)
    fired = set()
    for i in range(12):                                    # flickering artifact
        fired |= det.update(artifact if i % 2 else surface)
    assert fired == set()


def test_oblique_surface_press_still_fires():
    """Camera at an angle: a touching finger occludes surface farther along the
    ray, so it reads far 'higher' than its thickness (here 55mm). The slope-
    adaptive band ceiling must still accept it."""
    ramp = np.tile(np.linspace(800, 2400, W).astype(np.uint16), (H, 1))  # steep tilt
    det = _detector([_tile(1, 300, 100, 400, 300)], ramp)
    press = _press(ramp, 320, 150, 380, 250, mm=55)   # would miss a fixed 35mm cap
    det.update(ramp)
    fired = set()
    for _ in range(5):
        fired |= det.update(press)
    assert fired == {1}


def test_noisy_zone_flicker_does_not_fire_after_calibration():
    """Noisy pixels get stricter per-pixel thresholds instead of firing (and the
    zone is NOT blanket-masked)."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    hi, lo = surface.copy(), _press(surface, 60, 60, 140, 140, mm=24)
    hi[60:140, 60:140] = 1012                            # flicker 1012 <-> 976
    det.calibrate_artifacts([hi, lo, hi, lo, hi, lo])   # std ~18mm in the region
    fired = set()
    for i in range(16):                                  # keep flickering ±12mm
        fired |= det.update(hi if i % 2 else lo)
    assert fired == set()


def test_whole_frame_shift_rejected_as_blob():
    """A global shift (camera nudge) moves everything into the band at once —
    one huge blob, not a finger -> rejected instead of firing every tile."""
    surface = _flat()
    tiles = [_tile(i, (i - 1) * 60 + 10, 50, (i - 1) * 60 + 60, 150) for i in range(1, 6)]
    det = _detector(tiles, surface)
    det.update(surface)
    shifted = _flat(985)  # whole frame 15mm closer, with motion
    fired = set()
    for _ in range(6):
        fired |= det.update(shifted)
    assert fired == set()


def test_refire_requires_release_and_quiet():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    press = _press(surface, 70, 70, 130, 130)
    det.update(surface)
    first = set()
    for _ in range(4):
        first |= det.update(press)
    assert first == {1}
    for _ in range(10):         # release + quiet period
        det.update(surface)
    again = set()
    for _ in range(4):          # second press fires again
        again |= det.update(press)
    assert again == {1}


def test_video_tracker_edge_trigger_and_release():
    t = DepthHitTracker(release_frames=2)
    assert t.update({1}) == {1}
    assert t.update({1}) == set()
    assert t.update(set()) == set()
    assert t.update(set()) == set()
    assert t.update({1}) == {1}
