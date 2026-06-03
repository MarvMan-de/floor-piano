"""Camera-free end-to-end demo of the full pipeline.

Drives the real warp -> detect -> audio path with synthetic depth frames, so you
can verify (and HEAR) the piano on the Pi WITHOUT an Orbbec camera attached.
A synthetic 'foot' sweeps across the 7 keys — you should hear C D E F G A B in order.

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
from audio import PianoAudio
from depth_camera import MockDepthCamera
from detection import sweep_frames
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
        "keys": list(constants.DEFAULT_KEYS),
        "floor_depth": 1000,
        "trigger_threshold": constants.DEFAULT_TRIGGER_THRESHOLD,
    }

    frames = sweep_frames(h, w, len(config["keys"]), floor_depth=1000, foot_depth=800)
    piano = FloorPiano(config,
                       camera=MockDepthCamera(frames),
                       audio=PianoAudio(keys=config["keys"]))

    log.info("Playing a synthetic foot sweep across %d keys (no camera) ...", len(config["keys"]))
    try:
        for frame in frames:
            _, active = piano.process_frame(frame)
            if active:
                log.info("trigger: %s", sorted(active))
            time.sleep(0.12)
    finally:
        piano.shutdown()
    log.info("Done. If you heard the notes, the detection + audio pipeline works.")


if __name__ == "__main__":
    main()
