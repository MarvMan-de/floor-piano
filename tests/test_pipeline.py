"""End-to-end FloorPiano regression tests with injected camera + audio.

These cover the runtime path main.py actually executes (warp -> blob detect ->
tracker -> audio edge-trigger) — the level where the boundary double-trigger
and flicker retrigger bugs lived.
"""
import numpy as np
import pytest

import constants
import detection as d
from depth_camera import DepthCameraError, MockDepthCamera
from main import FloorPiano

W, H = constants.TARGET_WIDTH, constants.TARGET_HEIGHT
FLOOR, TH = 1000, 50


class RecordingAudio:
    """Audio sink that records every note-on edge, like PianoAudio would play them."""

    def __init__(self):
        self.active = set()
        self.played = []

    def update(self, current):
        self.played.extend(sorted(d.newly_pressed(current, self.active)))
        self.active = set(current)

    def close(self):
        pass


def make_piano(frames=(), **overrides):
    config = {
        # identity-ish warp: source frame == canvas
        "corners": [[0, 0], [W, 0], [W, H], [0, H]],
        "num_white_keys": 14,
        "start_octave": 4,
        "floor_depth": FLOOR,
        "trigger_threshold": TH,
    }
    config.update(overrides)
    audio = RecordingAudio()
    piano = FloorPiano(config, camera=MockDepthCamera(list(frames)), audio=audio)
    return piano, audio


def foot_frame(x0, x1, y0=150, y1=190, depth=800):
    f = d.flat_floor_frame(H, W, FLOOR)
    f[y0:y1, x0:x1] = depth
    return f


def test_boundary_foot_plays_exactly_one_note():
    piano, audio = make_piano()
    # one foot, 70/30 across the G4|A4 boundary at x=500
    piano.process_frame(foot_frame(430, 530))
    assert audio.played == ["G4"]


def test_flicker_does_not_retrigger():
    piano, audio = make_piano()
    on, off = foot_frame(430, 480), d.flat_floor_frame(H, W, FLOOR)
    for f in (on, off, on, off, on):       # noisy mask: key drops out for 1 frame
        piano.process_frame(f)
    assert audio.played == ["G4"]          # ONE note, not three


def test_release_then_press_again_replays():
    piano, audio = make_piano()
    on, off = foot_frame(430, 480), d.flat_floor_frame(H, W, FLOOR)
    piano.process_frame(on)
    for _ in range(constants.RELEASE_FRAMES):
        piano.process_frame(off)           # full release
    piano.process_frame(on)
    assert audio.played == ["G4", "G4"]


def test_two_feet_chord_including_black_over_white():
    piano, audio = make_piano()
    keys = piano.keyboard
    bk = next(k for k in keys if k.name == "C#5")
    f = d.flat_floor_frame(H, W, FLOOR)
    f[bk.y0:bk.y1, bk.x0:bk.x1] = 800                 # foot on black C#5
    f[160:195, keys[7].x0 + 10:keys[7].x1 - 10] = 800  # foot on white C5 below it
    _, active = piano.process_frame(f)
    assert active == {"C#5", "C5"}


def test_high_object_over_mat_is_silent():
    piano, audio = make_piano()
    piano.process_frame(foot_frame(430, 530, depth=FLOOR - constants.MAX_PRESS_HEIGHT - 100))
    assert audio.played == []


def test_run_raises_on_camera_stall(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    piano, audio = make_piano(frames=[])   # camera immediately returns None forever
    piano.camera._loop = False

    class NoneForever:
        def start(self):
            return self

        def read_depth(self):
            return None

        def stop(self):
            pass

    piano.camera = NoneForever()
    with pytest.raises(DepthCameraError, match="no depth frames"):
        piano.run()


def test_corners_rescaled_when_depth_resolution_differs():
    piano, _ = make_piano(canvas_size=[W, H])
    M_before = piano.M.copy()
    piano._check_corners_fit(np.zeros((H // 2, W // 2), dtype=np.uint16))
    assert not np.allclose(piano.M, M_before)   # warp rebuilt for the smaller frame
    # rescaled corners now span the depth frame exactly: warping a full-frame
    # foot must cover the whole canvas again
    f = np.full((H // 2, W // 2), 800, dtype=np.uint16)
    warped = piano.warp(f)
    assert int((warped == 800).mean() * 100) >= 99


def test_hit_threshold_rescaled_with_corners():
    """After the depth-resolution corner rescale, the foot-size gate must keep
    meaning MIN_HIT_PIXELS *source* pixels (4x canvas px at half resolution)."""
    piano, audio = make_piano(canvas_size=[W, H])
    assert piano.min_hit_pixels == constants.MIN_HIT_PIXELS
    piano._check_corners_fit(np.zeros((H // 2, W // 2), dtype=np.uint16))
    assert piano.min_hit_pixels == pytest.approx(4 * constants.MIN_HIT_PIXELS, rel=0.05)
    # a 50-source-px noise blob (1/3 of the documented gate) must NOT fire
    f = np.full((H // 2, W // 2), FLOOR, dtype=np.uint16)
    f[80:85, 230:240] = 800
    piano.process_frame(f)
    assert audio.played == []


def test_min_hit_pixels_scales_with_warp_magnification():
    # mat quad = whole canvas -> scale 1.0 -> unchanged
    piano1, _ = make_piano()
    assert piano1.min_hit_pixels == constants.MIN_HIT_PIXELS
    # mat quad covers only 1/4 of the canvas area -> each source px becomes ~4
    # canvas px -> threshold must grow accordingly
    piano2, _ = make_piano(corners=[[0, 0], [W / 2, 0], [W / 2, H / 2], [0, H / 2]])
    assert piano2.min_hit_pixels == pytest.approx(4 * constants.MIN_HIT_PIXELS, rel=0.05)


def test_sigterm_before_run_prevents_loop(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    piano, audio = make_piano(frames=[foot_frame(430, 480)] * 1000)
    piano.stop()                            # SIGTERM arrives before run()
    piano.run()                             # must return immediately
    assert audio.played == []
