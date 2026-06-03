"""Tests for the camera-free demo building blocks: sweep_frames + MockDepthCamera."""
import detection as d
from depth_camera import MockDepthCamera


def test_sweep_frames_count_and_shape():
    h, w, n = 50, 700, 7
    frames = d.sweep_frames(h, w, n, floor_depth=1000, foot_depth=800, hold=3, gap=2)
    assert len(frames) == n * (3 + 2)
    assert all(f.shape == (h, w) for f in frames)


def test_sweep_frames_each_key_triggers_in_turn():
    h, w, n = 50, 700, 7
    frames = d.sweep_frames(h, w, n, floor_depth=1000, foot_depth=800, hold=3, gap=2)
    # The first frame of each key-block holds that key's foot.
    triggered = []
    for k in range(n):
        first_hold_frame = frames[k * 5]          # hold(3) + gap(2) = 5 per key
        hits = d.detect_hits(first_hold_frame, n, 1000, 50)
        triggered.append(hits)
    assert triggered == [{k} for k in range(n)]


def test_sweep_frames_gap_frames_have_no_hits():
    h, w, n = 50, 700, 7
    frames = d.sweep_frames(h, w, n, floor_depth=1000, foot_depth=800, hold=3, gap=2)
    gap_frame = frames[3]  # 4th frame of the first block is a gap (empty floor)
    assert d.detect_hits(gap_frame, n, 1000, 50) == set()


def test_mock_depth_camera_serves_then_stops():
    frames = [d.flat_floor_frame(10, 700, 1000), d.flat_floor_frame(10, 700, 900)]
    cam = MockDepthCamera(frames, loop=False)
    cam.start()
    assert int(cam.read_depth()[0, 0]) == 1000
    assert int(cam.read_depth()[0, 0]) == 900
    assert cam.read_depth() is None      # exhausted, no loop
    cam.stop()


def test_mock_depth_camera_loops():
    frames = [d.flat_floor_frame(10, 700, 1000)]
    cam = MockDepthCamera(frames, loop=True)
    cam.start()
    assert cam.read_depth() is not None
    assert cam.read_depth() is not None   # wraps around
