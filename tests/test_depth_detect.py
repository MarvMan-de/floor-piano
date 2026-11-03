"""Tests for webui/depth_detect.py — synthetic depth maps, no hardware.

The PressDetector must satisfy: nothing moves -> nothing fires (motion gate);
static artifacts near the surface never fire (artifact mask); a real press
(appears + sits in the contact band) fires exactly once until released.
"""
import numpy as np

from webui.depth_detect import (DepthHitTracker, PressDetector,
                               build_tile_label_map, fit_inverse_depth_plane,
                               plane_depth_map)

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


def _press(surface, x0, y0, x1, y1, mm=20, limb=True):
    """A press blob mm above the surface — plus, like every real press, the
    finger/foot it belongs to sticking up ABOVE the band right next to it
    (the attachment gate requires that). limb=False makes an orphan blob."""
    f = surface.copy()
    f[y0:y1, x0:x1] = surface[y0:y1, x0:x1] - mm
    if limb:
        f[y0:y1, x1:x1 + 10] = np.maximum(
            surface[y0:y1, x1:x1 + 10].astype(np.int32) - 300, 1).astype(f.dtype)
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


def test_orphan_shadow_wobble_does_not_fire():
    """Depth wobble under a moving sunlight shadow: an in-band, moving blob
    lying flat on the surface with NOTHING sticking up next to it. Must stay
    silent (this played keys on the real mat)."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    det.update(surface)
    fired = set()
    for i in range(8):
        # wobble flickers within the band, shaped like a shadow patch
        wob = _press(surface, 60, 60, 140, 140, mm=18 + (i % 3) * 4, limb=False)
        fired |= det.update(wob)
    assert fired == set()


def test_press_at_frame_border_fires_without_limb():
    """A foot entering from the frame edge: its leg is outside the view, so
    the attachment gate must not demand hover pixels."""
    surface = _flat()
    det = _detector([_tile(1, 0, 50, 100, 150)], surface)
    press = _press(surface, 0, 60, 60, 140, limb=False)  # touches x=0 border
    det.update(surface)
    fired = set()
    for _ in range(5):
        fired |= det.update(press)
    assert fired == {1}


def test_render_debug_marks_contact_red():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    press = _press(surface, 70, 70, 130, 130)
    det.update(surface)
    for _ in range(4):
        det.update(press)
    img = det.render_debug(press)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    # press pixels red (band+stable+motion+ring guard) …
    assert (img[100, 100] == [0, 0, 255]).all()
    # … hovering limb keeps its distance colormap (not red, not black)
    assert not (img[100, 135] == [0, 0, 255]).all()
    assert img[100, 135].sum() > 60


def test_standing_foot_edge_ring_does_not_fire():
    """A foot STANDING next to a key: its sloped sides always cut through the
    contact band as a thin ring around the hovering foot top. That ring grazed
    neighbouring tiles and played them — the ring guard must keep it silent."""
    surface = _flat()
    det = _detector([_tile(1, 180, 80, 280, 220)], surface)
    foot = surface.copy()
    foot[94:206, 194:266] = 1000 - 20    # 6px-thick in-band ring ...
    foot[100:200, 200:260] = 1000 - 80   # ... around the hovering foot top
    det.update(surface)
    fired = set()
    for _ in range(8):
        fired |= det.update(foot)
    assert fired == set()


def test_toe_press_fires_tile_under_toe_centre():
    """Toe pressing the next key while the rest of the foot hovers: the tile
    under the CONTACT CENTRE fires — not the one the foot's rim touches."""
    surface = _flat()
    tiles = [_tile(1, 90, 90, 165, 170),    # under the hovering foot
             _tile(2, 165, 90, 240, 170)]   # under the toe
    det = _detector(tiles, surface)
    step = surface.copy()
    step[100:160, 100:160] = 1000 - 100     # foot/heel hovering (silent)
    step[110:150, 160:200] = 1000 - 20      # toe in contact on tile 2
    det.update(surface)
    fired = set()
    for _ in range(6):
        fired |= det.update(step)
    assert fired == {2}


def test_hover_100mm_on_slope_never_fires():
    """Regression: a foot hovering well above the mat sometimes fired on
    sloped/oblique zones, because the slope-adaptive band ceiling could grow
    to 150mm. The cap is now 80mm — 100mm hover must stay silent even where
    the slope-expanded ceiling used to accept it."""
    ramp = np.tile(np.linspace(800, 2400, W).astype(np.uint16), (H, 1))
    det = _detector([_tile(1, 300, 100, 400, 300)], ramp)
    hover = _press(ramp, 320, 150, 380, 250, mm=100)  # old cap: in band; now: out
    det.update(ramp)
    fired = set()
    for _ in range(6):
        fired |= det.update(hover)
    assert fired == set()


def test_render_semantic_shows_tiles_foot_and_contact():
    from webui.depth_detect import render_semantic
    tiles = [{"id": 1, "label": "C4", "note": "C4", "enabled": True,
              "color": "#4A90D9",
              "polygon": [[50, 50], [150, 50], [150, 150], [50, 150]]},
             {"id": 2, "label": "D4", "note": "D4", "enabled": True,
              "color": "#7ED321",
              "polygon": [[150, 50], [250, 50], [250, 150], [150, 150]]}]
    hover = np.zeros((H, W), bool); hover[80:120, 300:360] = True
    contact = np.zeros((H, W), bool); contact[90:110, 170:230] = True
    img = render_semantic(tiles, W, H, hover=hover, contact=contact, held={2})
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert (img[95, 180] == [0, 0, 255]).all()           # contact patch red
    assert (img[100, 200] == [255, 255, 255]).all()      # centre dot white
    assert img[100, 330][2] > 150 and img[100, 330][0] < 120  # foot orange-ish
    # held tile 2 renders brighter than idle tile 1 (outside the contact patch)
    assert int(img[60, 200].sum()) > int(img[60, 100].sum())
    # background stays dark
    assert img[400, 500].sum() < 100


# ── foot calibration: measure the shoe-top height while standing ────────────

def _standing_feet(surface, top_mm=50):
    """Player standing still: two foot-sized objects whose TOP reads top_mm
    above the surface (the sole touches — that's what standing means), plus a
    bit of shin sticking up."""
    f = surface.copy()
    for x0 in (150, 350):
        f[200:300, x0:x0 + 60] = surface[200:300, x0:x0 + 60] - top_mm  # foot top
        f[170:200, x0 + 15:x0 + 45] = np.maximum(
            surface[170:200, x0 + 15:x0 + 45].astype(np.int32) - 350, 1
        ).astype(f.dtype)                                               # shin
    return f


def test_calibrate_touch_sets_ceiling_from_feet():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    assert det.contact_max == 35  # default before calibration
    stand = _standing_feet(surface, top_mm=50)
    mm = det.calibrate_touch([stand] * 5)
    assert mm is not None and 55 <= mm <= 75, mm   # ~50 + margin
    assert det.contact_max == mm
    assert int(det.max_map[10, 10]) == mm          # flat zone ceiling follows


def test_calibrate_touch_enables_thick_shoe_press():
    """A sneaker toe reads ~50mm above the mat when pressing — the default
    35mm ceiling missed it. After standing calibration it fires."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    det.calibrate_touch([_standing_feet(surface, top_mm=50)] * 5)
    press = _press(surface, 70, 70, 130, 130, mm=50)   # shoe-top at 50mm
    det.update(surface)
    fired = set()
    for _ in range(5):
        fired |= det.update(press)
    assert fired == {1}


def test_calibrate_touch_none_without_feet():
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    assert det.calibrate_touch([surface.copy()] * 5) is None
    assert det.contact_max == 35   # unchanged


def test_held_key_survives_motion_timeout_and_liftoff_fires_once():
    """Regression: every key sounded TWICE. A resting foot goes motion-silent
    after motion_hold frames -> the tile released + re-armed while the foot
    was still on it, and the lift-off pass back through the band fired it
    again. Held tiles must stay held while the foot is present, and only
    release + re-arm once the area is clear."""
    surface = _flat()
    det = _detector([_tile(1, 50, 50, 150, 150)], surface)
    press = _press(surface, 70, 70, 130, 130)
    lift = _press(surface, 70, 70, 130, 130, mm=60)   # foot rising: above band
    det.update(surface)
    fired = set()
    for _ in range(30):                 # long hold — far beyond motion_hold
        fired |= det.update(press)
    assert fired == {1}
    assert det.held == {1}              # STILL held despite motion silence
    for _ in range(3):                  # lift-off passes back through the band
        fired |= det.update(_press(surface, 70, 70, 130, 130, mm=30))
    for _ in range(3):
        fired |= det.update(lift)
    for _ in range(6):                  # foot gone
        fired |= det.update(surface)
    assert fired == {1}                 # exactly ONE sound for the whole press
    # a genuine second press afterwards still works
    again = set()
    for _ in range(5):
        again |= det.update(press)
    assert again == {1}


# ── plane fallback ───────────────────────────────────────────────────────────

def _tilted(a=0.02, b=-0.27, c=0.77):
    """A real tilted plane: 1/z linear in the image coordinates (z ~1300-2000)."""
    u = np.arange(W)[None, :] / W
    v = np.arange(H)[:, None] / H
    return np.round(1000.0 / (a * u + b * v + c)).astype(np.uint16)


PLANE_TILES = [_tile(1, 50, 50, 250, 250), _tile(2, 300, 50, 500, 250)]


def _empty_frames(surface, holes=(), n=8, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = surface.astype(np.int32) + np.round(
            rng.normal(0, sigma, surface.shape)).astype(np.int32)
        for (y0, y1, x0, x1) in holes:
            f[y0:y1, x0:x1] = 0
        out.append(np.clip(f, 0, 65535).astype(np.uint16))
    return out


def _calibrated(frames, tiles=PLANE_TILES, **kw):
    ref = np.median(np.stack(frames), axis=0).astype(np.uint16)
    det = PressDetector(ref, **kw)
    det.set_tiles(*build_tile_label_map(tiles, W, H))
    det.calibrate_artifacts(frames)
    return det


def _foot(surface, y0, y1, x0, x1, mm=30):
    f = surface.astype(np.int32).copy()
    f[y0:y1, x0:x1] -= mm
    f[y0:y1, x1:x1 + 10] -= 300          # the limb above the band
    return f.astype(np.uint16)


def test_plane_fit_recovers_tilted_plane():
    surf = _tilted()
    rng = np.random.default_rng(1)
    noisy = surf.astype(np.int32) + np.round(rng.normal(0, 2, surf.shape)).astype(np.int32)
    mask = np.zeros(surf.shape, bool)
    mask[40:260, 40:510] = True
    fit = fit_inverse_depth_plane(noisy, mask)
    assert fit is not None and fit["inlier_ratio"] > 0.95 and fit["rms_mm"] < 3
    err = np.abs(plane_depth_map(fit["coef"], surf.shape).astype(int) - surf.astype(int))
    assert err[mask].max() <= 3
    # a linear-in-z fit would be visibly wrong on this tilt (design decision)
    vv, uu = np.mgrid[0:H, 0:W]
    A = np.stack([uu[mask] / W, vv[mask] / H, np.ones(mask.sum())], 1)
    c, *_ = np.linalg.lstsq(A, surf[mask].astype(float), rcond=None)
    assert np.abs(A @ c - surf[mask]).max() > 10


def test_holes_rescued_by_plane():
    surf = _tilted()
    holes = [(100, 140, 100, 140)]
    det = _calibrated(_empty_frames(surf, holes))
    r = det.rescued
    assert r is not None and r[110, 110] and not det.artifact[110, 110]
    assert abs(int(det.surface[110, 110]) - int(surf[110, 110])) <= 3
    assert det.min_map[110, 110] >= 20
    assert det.plane_info["ok"] and det.plane_info["rescued_frac"] > 0


def test_press_on_rescued_hole_fires_only_with_fallback():
    surf = _tilted()
    holes = [(100, 140, 100, 140)]
    frames = _empty_frames(surf, holes)
    press = _foot(surf, 105, 135, 105, 135)
    for fallback, expected in ((True, {1}), (False, set())):
        det = _calibrated(frames, plane_fallback=fallback)
        det.update(frames[0])
        det.update(frames[1])
        fired = set()
        for _ in range(6):
            fired |= det.update(press)
        assert fired == expected, fallback


def test_inband_garbage_flicker_stays_masked():
    surf = _tilted()
    rng = np.random.default_rng(3)
    frames = _empty_frames(surf)
    for f in frames:                       # reflection: 0 or 25 mm "in front"
        blk = f[150:190, 350:390].astype(np.int32)
        f[150:190, 350:390] = np.where(rng.random((40, 40)) < 0.5, 0, blk - 25)
    det = _calibrated(frames)
    assert det.artifact[170, 370]
    fired = set()
    det.update(frames[0])
    for i in range(12):
        fired |= det.update(frames[i % len(frames)])
    assert fired == set()


def test_reflection_reading_behind_plane_is_rescued():
    surf = _tilted()
    rng = np.random.default_rng(4)
    frames = _empty_frames(surf)
    for f in frames:                       # multipath: far too deep, noisy
        f[150:190, 350:390] = (surf[150:190, 350:390].astype(np.int32)
                               + 80 + rng.integers(0, 120, (40, 40))).astype(np.uint16)
    det = _calibrated(frames)
    assert det.rescued is not None and det.rescued[170, 370]
    assert abs(int(det.surface[170, 370]) - int(surf[170, 370])) <= 3


def test_live_zero_stays_blind():
    surf = _tilted()
    holes = [(100, 140, 100, 140)]
    frames = _empty_frames(surf, holes)
    det = _calibrated(frames)
    press = _foot(surf, 105, 135, 105, 135)
    press[105:135, 105:135] = 0            # sensor still blind while foot is down
    det.update(frames[0])
    fired = set()
    for _ in range(6):
        fired |= det.update(press)
    assert fired == set()


def test_reliable_bump_keeps_measured_surface():
    surf = _tilted().astype(np.int32)
    surf[120:160, 120:160] -= 25           # real, stable 25 mm bump in the mat
    surf = surf.astype(np.uint16)
    det = _calibrated(_empty_frames(surf, sigma=1.0))
    assert int(det.surface[140, 140]) == int(surf[140, 140]) or \
        abs(int(det.surface[140, 140]) - int(surf[140, 140])) <= 2
    press = _foot(surf, 125, 155, 125, 155)
    det.update(surf)
    fired = set()
    for _ in range(6):
        fired |= det.update(press)
    assert fired == {1}


def test_plane_fit_failure_falls_back_to_old_behaviour():
    surf = _flat(1000).copy()
    surf[:, 320:] = 1400                   # two-level step: no single plane
    frames = _empty_frames(surf, [(100, 140, 100, 140)])
    on = _calibrated(frames, tiles=[_tile(1, 0, 0, 640, 480)])
    off = _calibrated(frames, tiles=[_tile(1, 0, 0, 640, 480)], plane_fallback=False)
    assert not on.plane_info["ok"]
    for name in ("surface", "artifact", "min_map", "max_map", "motion_map"):
        assert np.array_equal(getattr(on, name), getattr(off, name)), name


def test_switch_off_is_identical_to_pre_feature_golden():
    """Hash recorded on the commit BEFORE the feature (scenario: tilted plane,
    holes + reflection flicker, one press)."""
    import hashlib
    rng = np.random.default_rng(0)
    surf = _tilted()
    frames = []
    for _ in range(8):
        f = surf.astype(np.int32) + np.round(rng.normal(0, 2, surf.shape)).astype(np.int32)
        f[100:140, 100:140] = 0
        f[150:190, 350:390] = np.where(rng.random((40, 40)) < .5, 0,
                                       f[150:190, 350:390] - 25)
        frames.append(np.clip(f, 0, 65535).astype(np.uint16))
    ref = np.median(np.stack(frames), axis=0).astype(np.uint16)
    det = PressDetector(ref, plane_fallback=False)
    det.set_tiles(*build_tile_label_map(PLANE_TILES, W, H))
    det.calibrate_artifacts(frames)
    press = surf.astype(np.int32).copy()
    press[200:240, 310:350] -= 20
    press[200:240, 350:360] -= 300
    press = press.astype(np.uint16)
    fired = [sorted(det.update(d)) for d in [frames[0]] * 3 + [press] * 6 + [frames[1]] * 6]
    h = hashlib.sha256()
    for a in (det.artifact, det.min_map, det.max_map, det.motion_map,
              det.render_debug(press)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(fired).encode())
    assert h.hexdigest() == "e9d944a13c1b753a9fec8de0d989b758a35aafee1e28c6b997c4dafd323c23ef"


def test_set_tiles_after_calibration_restricts_rescue():
    surf = _tilted()
    frames = _empty_frames(surf, [(100, 140, 100, 140), (400, 440, 560, 600)])
    ref = np.median(np.stack(frames), axis=0).astype(np.uint16)
    det = PressDetector(ref)
    det.calibrate_artifacts(frames)           # no tiles yet: hull domain
    before = det._maps
    det.set_tiles(*build_tile_label_map(PLANE_TILES, W, H))
    assert det._maps is not before            # swapped, not mutated
    assert det.rescued is not None and det.rescued[110, 110]
    assert not det.rescued[420, 580]          # outside the mat now


def test_render_debug_tints_rescued_magenta():
    surf = _tilted()
    det = _calibrated(_empty_frames(surf, [(100, 140, 100, 140)]))
    live = surf.copy()
    img = det.render_debug(live)
    px = img[110, 110].astype(int)            # B, G, R
    assert px[2] > px[1] and px[0] > px[1]
