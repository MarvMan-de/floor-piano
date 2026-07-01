import json
import logging
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

# Resolve sibling imports regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants
from depth_camera import DepthCamera, DepthCameraError
from detection import (above_floor_mask, build_keyboard, detect_hits_labeled,
                       keyboard_label_map, median_floor_depth,
                       suppress_white_under_black, validate_config)

log = logging.getLogger("floor_piano")

# Path to the WebUI tile config — watched for hot-reload without restart.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_PATH = os.path.join(_SCRIPT_DIR, "..", "config", "tiles.json")


class ConfigError(RuntimeError):
    """Raised when config.json is missing or invalid."""


def load_config(config_path):
    if not os.path.exists(config_path):
        raise ConfigError(f"{config_path} not found. Run 'python src/calibrate.py' first.")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        validate_config(cfg)
    except (json.JSONDecodeError, ValueError) as e:
        raise ConfigError(f"invalid config at {config_path}: {e}. Re-run calibrate.py.") from e
    return cfg


class FloorPiano:
    """The runtime piano: warp depth -> detect key hits -> play audio.

    Camera and audio are injected so the loop can be exercised with a
    MockDepthSource and a fake audio sink in tests, and so neither is created
    until the object is actually run on hardware.
    """

    def __init__(self, config, camera=None, audio=None):
        self.config = config
        self.floor_depth = int(config.get("floor_depth", constants.DEFAULT_FLOOR_DEPTH))
        self.threshold = int(config.get("trigger_threshold", constants.DEFAULT_TRIGGER_THRESHOLD))

        self.target_width = constants.TARGET_WIDTH
        self.target_height = constants.TARGET_HEIGHT
        corners = np.float32(config["corners"])
        dst = np.float32([[0, 0], [self.target_width, 0],
                          [self.target_width, self.target_height], [0, self.target_height]])
        self.M = cv2.getPerspectiveTransform(corners, dst)

        # Build the piano keyboard (14 white + 10 black by default) and a label map
        # so each frame's per-key hit count is a single vectorised bincount.
        num_white = int(config.get("num_white_keys", constants.DEFAULT_NUM_WHITE))
        start_octave = int(config.get("start_octave", constants.START_OCTAVE))
        self.keyboard = build_keyboard(num_white, self.target_width, self.target_height, start_octave)
        self.key_names = [k.name for k in self.keyboard]
        self.label_map = keyboard_label_map(self.keyboard, self.target_width, self.target_height)
        log.info("Keyboard: %d white + %d black = %d keys (%s..%s)",
                 num_white, len(self.keyboard) - num_white, len(self.keyboard),
                 self.keyboard[0].name, self.keyboard[num_white - 1].name)

        self.camera = camera if camera is not None else DepthCamera()
        if audio is not None:
            self.audio = audio
        else:
            from audio import PianoAudio  # deferred so an injected audio sink needs no pygame
            self.audio = PianoAudio(keys=self.key_names)
        self._running = False

        # WebUI tile config — populated by the watcher thread, used read-only in process_frame.
        self._tiles = []
        self._tiles_lock = threading.Lock()
        self._tiles_mtime = 0.0

    def warp(self, depth_array):
        # INTER_NEAREST: never blend in the 0 ("no reading") holes, which would
        # create artificially-close edge pixels and false triggers.
        return cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height),
                                   flags=cv2.INTER_NEAREST)

    def _check_corners_fit(self, depth_array):
        """Warn once if calibration corners fall outside the depth frame.

        The ArUco corners are detected in RGB space; if they don't fit the depth
        frame, the key mapping will be wrong (CODE_REVIEW #2, RGB->depth registration).
        """
        dh, dw = depth_array.shape[:2]
        if any(not (0 <= x <= dw and 0 <= y <= dh) for x, y in self.config["corners"]):
            log.warning("Calibration corners fall outside the depth frame (%dx%d). They were "
                        "detected in RGB space — keys will map wrong until RGB->depth "
                        "registration is fixed (see CODE_REVIEW #2).", dw, dh)

    def auto_level_floor(self, depth_array):
        """Re-sample the floor depth from the current scene (EMA-smoothed)."""
        new_floor = median_floor_depth(self.warp(depth_array))
        if new_floor is not None:
            a = constants.FLOOR_EMA_ALPHA
            self.floor_depth = int((1 - a) * self.floor_depth + a * new_floor)
            log.info("Re-leveled floor to %dmm", self.floor_depth)

    def process_frame(self, depth_array):
        """Pure-ish step: returns (warped_depth, set_of_active_note_names) and plays audio."""
        warped = self.warp(depth_array)
        active_idx = detect_hits_labeled(warped, self.label_map, len(self.keyboard),
                                         self.floor_depth, self.threshold)
        active_idx = suppress_white_under_black(active_idx, self.keyboard)
        active = {self.key_names[i] for i in active_idx}
        self.audio.update(active)
        return warped, active

    def stop(self):
        """Request a graceful shutdown (used by the SIGTERM handler)."""
        self._running = False

    def run(self):
        headless = not os.environ.get('DISPLAY')  # unset OR empty -> headless
        log.info("--- FLOOR PIANO ---  floor=%dmm  trigger<%dmm  mode=%s",
                 self.floor_depth, self.floor_depth - self.threshold,
                 "headless" if headless else "GUI")
        if not headless:
            log.info("Press 'q' to quit, 'r' to re-level the floor.")

        self.camera.start()
        self._running = True
        watcher = threading.Thread(target=self._watch_tiles, daemon=True, name="tiles-watcher")
        watcher.start()
        last_fps_log = 0.0
        corners_checked = False
        try:
            while self._running:
                t0 = time.time()
                depth = self.camera.read_depth()
                if depth is None:
                    continue
                if not corners_checked:
                    self._check_corners_fit(depth)
                    corners_checked = True

                warped, active = self.process_frame(depth)

                if not headless:
                    self._render(warped, active)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        log.info("Re-leveling floor...")
                        self.auto_level_floor(depth)

                now = time.time()
                if now - last_fps_log >= 2.0:
                    fps = 1.0 / max(now - t0, 1e-6)
                    log.info("FPS %.1f | floor %dmm | active %s", fps, self.floor_depth, sorted(active))
                    last_fps_log = now
        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.shutdown()

    def _render(self, warped, active):
        mask = above_floor_mask(warped, self.floor_depth, self.threshold)
        vis = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        # Above-floor pixels (feet) as a red overlay first, so the keys draw on top.
        vis[mask] = [0, 0, 80]
        # White keys (outlines), then black keys (filled) on top — like a real keyboard.
        for k in self.keyboard:
            if k.kind == "white":
                color = (0, 220, 0) if k.name in active else (90, 90, 90)
                cv2.rectangle(vis, (k.x0, 0), (k.x1, self.target_height), color, 2)
                cv2.putText(vis, k.name, (k.x0 + 3, self.target_height - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for k in self.keyboard:
            if k.kind == "black":
                color = (0, 255, 0) if k.name in active else (35, 35, 35)
                cv2.rectangle(vis, (k.x0, k.y0), (k.x1, k.y1), color, -1)
                cv2.rectangle(vis, (k.x0, k.y0), (k.x1, k.y1), (200, 200, 200), 1)
        cv2.imshow("Floor Piano - 3D View", vis)

    def _watch_tiles(self):
        """Poll config/tiles.json every 1 s and hot-reload on mtime change."""
        while self._running:
            try:
                stat = os.stat(TILES_PATH)
                if stat.st_mtime != self._tiles_mtime:
                    with open(TILES_PATH) as f:
                        doc = json.load(f)
                    tiles = [t for t in doc.get("tiles", []) if t.get("enabled", True)]
                    with self._tiles_lock:
                        self._tiles = tiles
                        self._tiles_mtime = stat.st_mtime
                    log.info("Tiles hot-reloaded: %d tile(s)", len(tiles))
            except FileNotFoundError:
                pass
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning("tiles.json parse error, keeping previous tiles: %s", e)
            except Exception as e:
                log.warning("Tile watcher error: %s", e)
            time.sleep(1.0)

    def shutdown(self):
        log.info("Shutting down...")
        self.camera.stop()
        try:
            self.audio.close()
        except Exception:
            pass
        if os.environ.get('DISPLAY'):
            cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    try:
        config = load_config(config_path)
    except ConfigError as e:
        log.error("%s", e)
        sys.exit(1)

    piano = FloorPiano(config)
    # Clean shutdown on `systemctl stop` (SIGTERM); Ctrl-C arrives as KeyboardInterrupt.
    signal.signal(signal.SIGTERM, lambda *_: piano.stop())

    try:
        piano.run()
    except DepthCameraError as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
