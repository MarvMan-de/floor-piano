"""Tests for the 24-key piano layout (14 white + 10 black) and labeled detection."""
import numpy as np

import constants
import detection as d

W, H = 1400, 200


def kb():
    return d.build_keyboard(constants.DEFAULT_NUM_WHITE, W, H)


# --- layout counts & names -------------------------------------------------

def test_default_keyboard_has_14_white_10_black():
    keys = kb()
    whites = [k for k in keys if k.kind == "white"]
    blacks = [k for k in keys if k.kind == "black"]
    assert len(whites) == 14
    assert len(blacks) == 10
    assert len(keys) == 24


def test_white_key_names_two_octaves():
    whites = [k.name for k in kb() if k.kind == "white"]
    assert whites == ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
                      "C5", "D5", "E5", "F5", "G5", "A5", "B5"]


def test_black_key_names_and_no_e_sharp_or_b_sharp():
    blacks = [k.name for k in kb() if k.kind == "black"]
    assert blacks == ["C#4", "D#4", "F#4", "G#4", "A#4",
                      "C#5", "D#5", "F#5", "G#5", "A#5"]
    assert not any(b.startswith("E#") or b.startswith("B#") for b in blacks)


def test_keyboard_note_names_matches_build_keyboard():
    names = d.keyboard_note_names(constants.DEFAULT_NUM_WHITE, constants.START_OCTAVE)
    assert len(names) == 24
    assert set(names) == {k.name for k in kb()}


# --- geometry --------------------------------------------------------------

def test_black_keys_are_top_only_and_narrower_than_white():
    keys = kb()
    white_w = W / 14
    for k in keys:
        if k.kind == "black":
            assert k.y0 == 0 and k.y1 < H               # top strip only
            assert (k.x1 - k.x0) < white_w              # narrower than a white key


def test_white_keys_tile_full_width():
    whites = [k for k in kb() if k.kind == "white"]
    assert whites[0].x0 == 0
    assert whites[-1].x1 == W
    for a, b in zip(whites, whites[1:]):
        assert a.x1 == b.x0                              # no gaps/overlap between whites


def test_label_map_black_overwrites_white():
    keys = kb()
    lm = d.keyboard_label_map(keys, W, H)
    assert lm.shape == (H, W)
    bi, bk = next((i, k) for i, k in enumerate(keys) if k.kind == "black")
    cx = (bk.x0 + bk.x1) // 2
    assert lm[1, cx] == bi                               # top strip belongs to the black key


# --- labeled detection -----------------------------------------------------

def test_foot_on_white_lower_area_triggers_only_that_white():
    keys = kb()
    lm = d.keyboard_label_map(keys, W, H)
    wi = 3  # F4
    wk = keys[wi]
    bh = max(k.y1 for k in keys if k.kind == "black")
    frame = d.flat_floor_frame(H, W, 1000)
    frame[bh:H, wk.x0:wk.x1] = 800                       # below the black keys -> pure white area
    assert d.detect_hits_labeled(frame, lm, len(keys), 1000, 50) == {wi}


def test_foot_on_black_key_triggers_only_that_black():
    keys = kb()
    lm = d.keyboard_label_map(keys, W, H)
    bi, bk = next((i, k) for i, k in enumerate(keys) if k.kind == "black")
    frame = d.stamp_key(d.flat_floor_frame(H, W, 1000), bk, 800)
    assert d.detect_hits_labeled(frame, lm, len(keys), 1000, 50) == {bi}


def test_suppress_white_under_black_removes_overlapping_whites():
    keys = kb()
    bi, bk = next((i, k) for i, k in enumerate(keys) if k.kind == "black")
    overlap = {i for i, k in enumerate(keys)
               if k.kind == "white" and k.x0 < bk.x1 and k.x1 > bk.x0}
    assert overlap                                       # the black does overlap some whites
    assert d.suppress_white_under_black({bi} | overlap, keys) == {bi}


def test_note_filename_sharps_become_s():
    assert d.note_filename("C#4") == "Cs4.wav"
    assert d.note_filename("C4") == "C4.wav"


def test_keyboard_sweep_triggers_each_key_exactly_once():
    """End-to-end of the camera-free demo logic: every white AND black key fires
    once, in order, with suppression applied (regression for the bug where the
    full-rectangle white stamp got swallowed by the overlapping black key)."""
    keys = kb()
    lm = d.keyboard_label_map(keys, W, H)
    names = [k.name for k in keys]
    frames = d.keyboard_sweep_frames(keys, W, H, floor_depth=1000, foot_depth=800)

    prev, sequence = set(), []
    for f in frames:
        idx = d.suppress_white_under_black(
            d.detect_hits_labeled(f, lm, len(keys), 1000, 50), keys)
        cur = {names[i] for i in idx}
        sequence.extend(sorted(d.newly_pressed(cur, prev)))
        prev = cur

    assert sequence == names            # each key once, in keyboard order
    assert len(sequence) == 24
