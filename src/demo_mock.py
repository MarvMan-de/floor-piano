"""Camera-free end-to-end demo of the full pipeline.

Drives the real warp -> detect -> audio path with synthetic depth frames, so you
can verify (and HEAR) the piano on the Pi WITHOUT an Orbbec camera attached.
A synthetic 'foot' visits each of the 24 keys (14 white + 10 black) in turn.

Run:
    python3 src/demo_mock.py

Needs: numpy, opencv (cv2), pygame, and the generated samples in src/sounds/.
This exercises everything except the camera/SDK and the RGB->depth registration.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants
from depth_camera import MockDepthCamera
from detection import keyboard_sweep_frames
from main import FloorPiano

log = logging.getLogger("demo")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    w, h = constants.TARGET_WIDTH, constants.TARGET_HEIGHT
    # Corners = the whole frame, so the perspective warp is an identity map and we
    # feed already-"warped" synthetic frames straight through.
    config = {
        "corners": [[0, 0], [w, 0], [w, h], [0, h]],
        "num_white_keys": constants.DEFAULT_NUM_WHITE,
        "start_octave": constants.START_OCTAVE,
        "floor_depth": 1000,
        "trigger_threshold": constants.DEFAULT_TRIGGER_THRESHOLD,
    }

    # FloorPiano builds the keyboard + audio itself; reuse its keyboard for the sweep.
    piano = FloorPiano(config, camera=MockDepthCamera([]))
    frames = keyboard_sweep_frames(piano.keyboard, w, h,
                                   floor_depth=config["floor_depth"], foot_depth=800)

    log.info("Playing a synthetic foot sweep across %d keys (no camera) ...", len(piano.keyboard))
    try:
        for frame in frames:
            _, active = piano.process_frame(frame)
            if active:
                log.info("trigger: %s", sorted(active))
            time.sleep(0.10)
    finally:
        piano.shutdown()
    log.info("Done. If you heard the notes, the detection + audio pipeline works.")


if __name__ == "__main__":
    main()
