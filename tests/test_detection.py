"""Unit tests for the hardware-free detection logic.

These run with synthetic depth frames (just numpy arrays), so no camera, no
Orbbec SDK and no audio device are required. This is the safety net that lets us
change the trigger logic and implement RANSAC/registration without a Pi attached.
"""
import numpy as np
import pytest

import detection as d

HEIGHT, WIDTH = 200, 700
NUM_KEYS = 7
FLOOR = 1000          # mm to the floor
THRESHOLD = 50        # 50mm safety buffer -> trigger_depth = 950


# --- key_width -------------------------------------------------------------

def test_key_width_basic():
    assert d.key_width(700, 7) == 100


def test_key_width_rejects_non_positive():
    with pytest.raises(ValueError):
        d.key_width(700, 0)


# --- detect_hits -----------------------------------------------------------

def test_flat_floor_has_no_hits():
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == set()


def test_foot_in_one_zone_triggers_exactly_that_key():
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, key_index=2, num_keys=NUM_KEYS, foot_depth=800)
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == {2}


def test_two_feet_trigger_two_keys():
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, 0, NUM_KEYS, 700)
    frame = d.stamp_foot(frame, 6, NUM_KEYS, 700)
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == {0, 6}


def test_object_within_safety_buffer_does_not_trigger():
    # 990mm is below the floor reading but inside the 50mm buffer (>950) -> ignored.
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, 3, NUM_KEYS, FLOOR - 10)
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == set()


def test_small_blob_below_pixel_threshold_is_ignored():
    # 1 row x 100 cols = 100 px, below the default 150-pixel min -> noise, no fire.
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, 1, NUM_KEYS, 700, rows=(0, 1))
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == set()


def test_blob_above_pixel_threshold_fires():
    # 2 rows x 100 cols = 200 px, above the 150-pixel min.
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, 1, NUM_KEYS, 700, rows=(0, 2))
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == {1}


def test_zero_pixels_are_invalid_never_hits():
    # 0 = "no reading"; must never be treated as a foot even though 0 < trigger_depth.
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD) == set()


def test_custom_min_hit_pixels_is_respected():
    frame = d.flat_floor_frame(HEIGHT, WIDTH, FLOOR)
    frame = d.stamp_foot(frame, 4, NUM_KEYS, 700, rows=(0, 1))  # 100 px
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD, min_hit_pixels=50) == {4}
    assert d.detect_hits(frame, NUM_KEYS, FLOOR, THRESHOLD, min_hit_pixels=150) == set()


# --- above_floor_mask ------------------------------------------------------

def test_above_floor_mask_excludes_zero_and_floor():
    frame = np.array([[0, 800, 1000, 990]], dtype=np.uint16)  # trigger_depth = 950
    mask = d.above_floor_mask(frame, FLOOR, THRESHOLD)
    assert mask.tolist() == [[False, True, False, False]]


# --- newly_pressed (edge trigger) ------------------------------------------

def test_newly_pressed_only_returns_new_notes():
    assert d.newly_pressed({"C", "E"}, {"C"}) == {"E"}


def test_held_note_does_not_retrigger():
    assert d.newly_pressed({"C"}, {"C"}) == set()


def test_released_note_is_not_new():
    assert d.newly_pressed(set(), {"C"}) == set()


# --- median_floor_depth ----------------------------------------------------

def test_median_floor_depth_of_flat_floor():
    frame = d.flat_floor_frame(HEIGHT, WIDTH, 1234)
    assert d.median_floor_depth(frame) == 1234.0


def test_median_floor_depth_ignores_zero_pixels():
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    frame[:, :350] = 1000  # plenty of valid pixels, rest are invalid zeros
    assert d.median_floor_depth(frame) == 1000.0


def test_median_floor_depth_returns_none_when_too_sparse():
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    frame[0, :10] = 1000  # only 10 valid pixels < default 1000 minimum
    assert d.median_floor_depth(frame) is None


# --- validate_config -------------------------------------------------------

def good_config():
    return {
        "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "keys": ["C", "D", "E"],
        "floor_depth": 1000,
        "trigger_threshold": 50,
    }


def test_validate_config_accepts_good_config():
    assert d.validate_config(good_config()) is True


@pytest.mark.parametrize("missing", ["corners", "keys", "floor_depth", "trigger_threshold"])
def test_validate_config_rejects_missing_key(missing):
    cfg = good_config()
    cfg.pop(missing)
    with pytest.raises(ValueError):
        d.validate_config(cfg)


def test_validate_config_rejects_wrong_corner_count():
    cfg = good_config()
    cfg["corners"] = [[0, 0], [1, 1]]
    with pytest.raises(ValueError):
        d.validate_config(cfg)


def test_validate_config_rejects_floor_not_above_threshold():
    cfg = good_config()
    cfg["floor_depth"] = 40  # <= trigger_threshold 50
    with pytest.raises(ValueError):
        d.validate_config(cfg)


def test_validate_config_rejects_empty_keys():
    cfg = good_config()
    cfg["keys"] = []
    with pytest.raises(ValueError):
        d.validate_config(cfg)


def test_validate_config_rejects_non_dict():
    with pytest.raises(ValueError):
        d.validate_config([1, 2, 3])


@pytest.mark.parametrize("field", ["floor_depth", "trigger_threshold"])
def test_validate_config_rejects_non_numeric(field):
    cfg = good_config()
    cfg[field] = "1000"  # string from a hand-edited config -> ValueError, not TypeError
    with pytest.raises(ValueError):
        d.validate_config(cfg)


def test_validate_config_rejects_bool_as_number():
    cfg = good_config()
    cfg["floor_depth"] = True  # bool is technically an int; must be rejected
    with pytest.raises(ValueError):
        d.validate_config(cfg)


# --- MockDepthSource -------------------------------------------------------

def test_mock_depth_source_replays_frames_in_order():
    frames = [d.flat_floor_frame(10, 700, 1000), d.flat_floor_frame(10, 700, 900)]
    src = d.MockDepthSource(frames)
    assert len(src) == 2
    out = list(src.frames())
    assert [int(f[0, 0]) for f in out] == [1000, 900]
