"""Blob-based hit detection + HitTracker debounce (the two core runtime fixes)."""
import numpy as np
import pytest

import constants
import detection as d

W, H = 1400, 200
FLOOR, TH = 1000, 50


def kb():
    return d.build_keyboard(constants.DEFAULT_NUM_WHITE, W, H)


def setup():
    keys = kb()
    return keys, d.keyboard_label_map(keys, W, H)


# --- detect_hits_blobs ------------------------------------------------------

def test_blob_straddling_two_whites_fires_only_the_majority_key():
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    # foot 70% on white 4 (G4: x 400..500), 30% on white 5 — old per-key counting
    # fired both (each side has > MIN_HIT_PIXELS)
    frame[150:190, 430:530] = 800
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH)
    assert hits == {4}


def test_two_separate_blobs_fire_two_keys_chord():
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[150:190, 120:180] = 800    # foot on D4 (white 1)
    frame[150:190, 920:980] = 800    # foot on B4... actually white 9 = E5
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH)
    assert hits == {1, 9}


def test_black_plus_white_lower_band_chord_survives():
    """Two feet: one on a black key, one on the white area straight below it.

    The old suppress_white_under_black step silenced the white; the blob path
    keeps both (they are separate blobs, each argmax-assigned to its own key).
    """
    keys, lm = setup()
    bi, bk = next((i, k) for i, k in enumerate(keys) if k.kind == "black")
    below = next(i for i, k in enumerate(keys)
                 if k.kind == "white" and k.x0 < bk.x1 and k.x1 > bk.x0)
    bh = max(k.y1 for k in keys if k.kind == "black")
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[bk.y0:bk.y1, bk.x0:bk.x1] = 800                       # foot on the black
    frame[bh + 10:H - 5, keys[below].x0 + 5:keys[below].x1 - 5] = 800  # foot below it
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH)
    assert hits == {bi, below}


def test_blob_smaller_than_min_pixels_is_ignored():
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[100:110, 450:460] = 800    # 100 px < default 150
    assert d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH) == set()


def test_blob_above_press_height_band_is_ignored():
    """A foot swinging high over the mat (or a knee/torso) must not trigger."""
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[150:190, 420:480] = FLOOR - constants.MAX_PRESS_HEIGHT - 50  # too high
    assert d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH) == set()
    frame2 = d.flat_floor_frame(H, W, FLOOR)
    frame2[150:190, 420:480] = FLOOR - 100                              # foot-height
    assert d.detect_hits_blobs(frame2, lm, len(keys), FLOOR, TH) == {4}


def test_fragmented_foot_is_one_blob_after_closing():
    """Depth holes split a foot into nearby fragments; closing must merge them
    so the foot doesn't fire two keys."""
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[150:190, 460:500] = 800
    frame[150:190, 503:543] = 800    # 3px gap (a depth-hole stripe)
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH)
    assert len(hits) == 1


def test_keyboard_sweep_with_blobs_triggers_each_key_exactly_once():
    """The 24-key sweep through the BLOB path (the one main.py actually runs)."""
    keys, lm = setup()
    names = [k.name for k in keys]
    frames = d.keyboard_sweep_frames(keys, W, H, floor_depth=FLOOR, foot_depth=800)
    prev, sequence = set(), []
    for f in frames:
        idx = d.detect_hits_blobs(f, lm, len(keys), FLOOR, TH)
        cur = {names[i] for i in idx}
        sequence.extend(sorted(d.newly_pressed(cur, prev)))
        prev = cur
    assert sequence == names


def test_sticky_assignment_stops_boundary_chatter():
    """A blob rocking ~50/50 on a key boundary must not alternate notes."""
    keys, lm = setup()
    first = None
    held = set()
    # split jitters around the x=500 G4|A4 boundary: 52/48, 48/52, 51/49, ...
    for shift in (2, -2, 1, -3, 2, 0, -1):
        frame = d.flat_floor_frame(H, W, FLOOR)
        frame[150:190, 450 + shift:550 + shift] = 800
        hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH, sticky=held)
        assert len(hits) == 1
        if first is None:
            first = hits
        assert hits == first              # assignment never flips
        held = hits


def test_sticky_does_not_block_a_real_move():
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[150:190, 520:620] = 800        # decisively on white 5 (x 500..600)
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH, sticky={4})
    assert hits == {5}


def test_sticky_with_two_held_keys_keeps_the_dominant_one():
    """Both boundary keys held (merged chord / release window): the blob must
    stay with its majority key, not get handed to the weaker held neighbour."""
    keys, lm = setup()
    frame = d.flat_floor_frame(H, W, FLOOR)
    frame[150:190, 445:545] = 800        # 55/45 in favour of white 4
    hits = d.detect_hits_blobs(frame, lm, len(keys), FLOOR, TH, sticky={4, 5})
    assert hits == {4}


# --- HitTracker --------------------------------------------------------------

def test_tracker_attack_is_immediate():
    t = d.HitTracker(release_frames=3)
    assert t.update({5}) == {5}


def test_tracker_bridges_single_frame_dropout():
    """A one-frame mask flicker must NOT release + retrigger the note."""
    t = d.HitTracker(release_frames=3)
    t.update({5})
    assert t.update(set()) == {5}     # dropout frame 1: still held
    assert t.update({5}) == {5}       # back -> no edge for the audio layer
    assert t._missing[5] == 0


def test_tracker_releases_after_n_consecutive_misses():
    t = d.HitTracker(release_frames=3)
    t.update({5})
    t.update(set())
    t.update(set())
    assert t.update(set()) == set()   # 3rd consecutive miss -> released


def test_tracker_retrigger_after_release_works():
    t = d.HitTracker(release_frames=2)
    t.update({5})
    t.update(set())
    t.update(set())                   # released
    assert t.update({5}) == {5}       # pressing again is a fresh note


def test_tracker_independent_keys():
    t = d.HitTracker(release_frames=2)
    t.update({1, 2})
    assert t.update({2}) == {1, 2}    # 1 missing once -> still held
    assert t.update({2}) == {2}       # 1 missing twice -> released, 2 held


# --- hardened validate_config -----------------------------------------------

def good_config():
    return {
        "corners": [[0, 0], [640, 0], [640, 480], [0, 480]],
        "num_white_keys": 14,
        "floor_depth": 1000,
        "trigger_threshold": 50,
    }


def test_validate_config_rejects_swapped_corners():
    cfg = good_config()
    cfg["corners"] = [cfg["corners"][1], cfg["corners"][0],
                      cfg["corners"][3], cfg["corners"][2]]  # mirrored winding
    with pytest.raises(ValueError, match="quad"):
        d.validate_config(cfg)


def test_validate_config_rejects_degenerate_corners():
    cfg = good_config()
    cfg["corners"] = [[0, 0], [0, 0], [0, 0], [0, 0]]
    with pytest.raises(ValueError, match="quad"):
        d.validate_config(cfg)


def test_validate_config_rejects_non_numeric_corner():
    cfg = good_config()
    cfg["corners"][2] = [640, "480"]
    with pytest.raises(ValueError, match="corner"):
        d.validate_config(cfg)


def test_validate_config_rejects_nan_corner():
    cfg = good_config()
    cfg["corners"][1] = [float("nan"), 0]
    with pytest.raises(ValueError, match="corner"):
        d.validate_config(cfg)


def test_validate_config_rejects_bad_start_octave():
    cfg = good_config()
    cfg["start_octave"] = "4"
    with pytest.raises(ValueError, match="start_octave"):
        d.validate_config(cfg)


# --- sample_floor_depth returns None on failure ------------------------------

def test_sample_floor_depth_returns_none_on_edge_or_invalid():
    depth = np.zeros((100, 100), dtype=np.uint16)
    # centre of an all-invalid frame
    assert d.sample_floor_depth(depth, (50, 50), (100, 100)) is None
    # point at the frame edge
    depth[:] = 1000
    assert d.sample_floor_depth(depth, (1, 1), (100, 100)) is None


def test_sample_floor_depth_returns_median_when_valid():
    depth = np.full((100, 100), 1234, dtype=np.uint16)
    assert d.sample_floor_depth(depth, (50, 50), (100, 100)) == 1234
