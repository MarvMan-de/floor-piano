"""Run the full pipeline against an MP4 (or any video file) — no camera needed.

Feeds a recorded video through the real FloorPiano (warp -> label -> detect ->
audio), turning each frame into a depth-like image (see VideoDepthCamera). Lets
you test the piano from a clip on a laptop or the Pi without an Orbbec camera.

    python3 src/demo_video.py clip.mp4
    python3 src/demo_video.py clip.mp4 --invert --loop --show
    python3 src/demo_video.py clip.mp4 --floor 1000 --near 700 --threshold 50

A normal video has no real depth, so brightness is mapped to millimetres: by
default a DARK object counts as "close" (a foot) and bright as the floor. If
nothing triggers, try --invert, or pull --near further below --floor.

Exercises everything except the camera/SDK and the RGB->depth registration.
Needs: numpy, opencv (cv2), and (unless --no-audio) pygame + the samples in
src/sounds/.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2

import constants
from depth_camera import DepthCameraError, VideoDepthCamera
from main import FloorPiano

log = logging.getLogger("demo_video")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test the floor piano against a video file.")
    p.add_argument("video", help="path to an mp4 / video file")
    p.add_argument("--white", type=int, default=constants.DEFAULT_NUM_WHITE,
                   help="number of white keys (black keys are derived)")
    p.add_argument("--octave", type=int, default=constants.START_OCTAVE,
                   help="octave of the leftmost white key")
    p.add_argument("--floor", type=int, default=constants.DEFAULT_FLOOR_DEPTH,
                   help="depth (mm) the brightest pixels map to (the 'floor')")
    p.add_argument("--near", type=int, default=None,
                   help="depth (mm) the darkest pixels map to (default: floor-200)")
    p.add_argument("--threshold", type=int, default=constants.DEFAULT_TRIGGER_THRESHOLD,
                   help="trigger buffer (mm) above the floor")
    p.add_argument("--invert", action="store_true",
                   help="treat BRIGHT pixels as close instead of dark ones")
    p.add_argument("--loop", action="store_true", help="replay the video on repeat")
    p.add_argument("--fps", type=float, default=0.0,
                   help="cap playback speed (0 = as fast as possible)")
    p.add_argument("--no-audio", action="store_true", help="run silently (no pygame)")
    p.add_argument("--show", action="store_true",
                   help="show the warped detection view (needs a display)")
    return p.parse_args(argv)


class _SilentAudio:
    """No-op audio sink so --no-audio works without pygame/samples."""
    def update(self, active):
        pass

    def close(self):
        pass


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    camera = VideoDepthCamera(args.video, floor_depth=args.floor, near_depth=args.near,
                              invert=args.invert, loop=args.loop)
    try:
        camera.start()
    except DepthCameraError as e:
        log.error("%s", e)
        sys.exit(1)

    # Corners = the whole video frame, so the perspective warp just resamples the
    # clip onto the piano canvas (a plain video has no ArUco markers to detect).
    w, h = camera.width, camera.height
    config = {
        "corners": [[0, 0], [w, 0], [w, h], [0, h]],
        "num_white_keys": args.white,
        "start_octave": args.octave,
        "floor_depth": args.floor,
        "trigger_threshold": args.threshold,
    }
    audio = _SilentAudio() if args.no_audio else None  # None -> FloorPiano builds PianoAudio
    piano = FloorPiano(config, camera=camera, audio=audio)

    headless = args.show and not os.environ.get("DISPLAY")
    if headless:
        log.warning("--show needs a display ($DISPLAY unset); running without the window.")
    show = args.show and not headless

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    log.info("Playing %s (floor=%dmm near=%dmm invert=%s) ...",
             args.video, args.floor, camera.near_depth, args.invert)
    n = 0
    try:
        while True:
            t0 = time.time()
            depth = camera.read_depth()
            if depth is None:
                break  # end of video (loop=False)
            n += 1
            warped, active = piano.process_frame(depth)
            if active:
                log.info("frame %d trigger: %s", n, sorted(active))
            if show:
                piano._render(warped, active)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            if period:
                time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        piano.shutdown()
    log.info("Done after %d frames.", n)


if __name__ == "__main__":
    main()
