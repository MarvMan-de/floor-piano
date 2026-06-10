import json
import logging
import os
import signal
import sys
import time

import cv2
import numpy as np

# Resolve sibling imports regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants
from depth_camera import DepthCamera, DepthCameraError
from detection import (HitTracker, above_floor_mask, build_keyboard,
                       detect_hits_blobs, keyboard_label_map, median_floor_depth,
                       validate_config)

log = logging.getLogger("floor_piano")


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
        self.max_press_height = int(config.get("max_press_height", constants.MAX_PRESS_HEIGHT))

        self.target_width = constants.TARGET_WIDTH
        self.target_height = constants.TARGET_HEIGHT
        self._build_warp(config["corners"])
        self._set_hit_threshold(config["corners"])

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

        # Debounce: a key releases only after RELEASE_FRAMES consecutive misses,
        # so depth noise can't retrigger a held note frame after frame.
        self.tracker = HitTracker(int(config.get("release_frames", constants.RELEASE_FRAMES)))

        self.camera = camera if camera is not None else DepthCamera()
        if audio is not None:
            self.audio = audio
        else:
            from audio import PianoAudio  # deferred so an injected audio sink needs no pygame
            self.audio = PianoAudio(keys=self.key_names)
        self._running = False
        self._stop_requested = False

    def _build_warp(self, corners):
        dst = np.float32([[0, 0], [self.target_width, 0],
                          [self.target_width, self.target_height], [0, self.target_height]])
        self.M = cv2.getPerspectiveTransform(np.float32(corners), dst)

    def _set_hit_threshold(self, corners):
        """Convert MIN_HIT_PIXELS ("foot-sized in SOURCE pixels") to canvas pixels.

        The warp rescales the mat onto the fixed canvas, so the threshold must
        scale with the warp magnification — otherwise sensitivity would change
        with camera resolution and mounting height. Recomputed whenever the
        warp corners change (see _check_corners_fit).
        """
        quad_area = abs(sum(corners[i][0] * corners[(i + 1) % 4][1]
                            - corners[(i + 1) % 4][0] * corners[i][1]
                            for i in range(4))) / 2.0
        scale = self.target_width * self.target_height / max(quad_area, 1.0)
        self.min_hit_pixels = int(min(max(constants.MIN_HIT_PIXELS * scale, 50), 5000))
        log.info("Hit threshold: %d canvas px (%d source px, warp scale %.2fx)",
                 self.min_hit_pixels, constants.MIN_HIT_PIXELS, scale)

    def warp(self, depth_array):
        # INTER_NEAREST: never blend in the 0 ("no reading") holes, which would
        # create artificially-close edge pixels and false triggers.
        return cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height),
                                   flags=cv2.INTER_NEAREST)

    def _check_corners_fit(self, depth_array):
        """Adapt RGB-space corners to the depth frame once its size is known.

        The ArUco corners are detected in RGB space. When the depth stream has a
        different resolution, rescale the corners (config 'canvas_size' = the RGB
        size they were measured in) and rebuild the warp — otherwise the key grid
        lands on the wrong part of the mat. This fixes the RESOLUTION part of the
        mismatch only; the RGB/depth FOV offset still needs D2C alignment
        (CODE_REVIEW #2), so a warning remains if corners fall outside the frame.
        """
        dh, dw = depth_array.shape[:2]
        canvas = self.config.get("canvas_size")
        if canvas and (int(canvas[0]) != dw or int(canvas[1]) != dh):
            rw, rh = float(canvas[0]), float(canvas[1])
            scaled = [[x * dw / rw, y * dh / rh] for x, y in self.config["corners"]]
            self._build_warp(scaled)
            self._set_hit_threshold(scaled)  # foot size changes with the source scale too
            log.info("Corners rescaled from RGB %dx%d to depth %dx%d space.",
                     int(rw), int(rh), dw, dh)
            corners = scaled
        else:
            corners = self.config["corners"]
        if any(not (0 <= x <= dw and 0 <= y <= dh) for x, y in corners):
            log.warning("Calibration corners fall outside the depth frame (%dx%d). They were "
                        "detected in RGB space — keys will map wrong until RGB->depth "
                        "registration is fixed (see CODE_REVIEW #2).", dw, dh)

    def auto_level_floor(self, depth_array):
        """Re-sample the floor depth from the current scene (EMA-smoothed).

        Pixels currently above the floor (someone standing on the mat) are
        masked out first, so a re-level mid-play doesn't drag the floor up.
        """
        warped = self.warp(depth_array)
        floor_only = warped.copy()
        floor_only[above_floor_mask(warped, self.floor_depth, self.threshold)] = 0
        new_floor = median_floor_depth(floor_only)
        if new_floor is not None:
            a = constants.FLOOR_EMA_ALPHA
            self.floor_depth = int(round((1 - a) * self.floor_depth + a * new_floor))
            log.info("Re-leveled floor to %dmm", self.floor_depth)

    def process_frame(self, depth_array):
        """Pure-ish step: returns (warped_depth, set_of_active_note_names) and plays audio.

        Detection is blob-based: each above-floor blob (one foot) presses exactly
        one key, so a foot straddling a key boundary no longer fires both keys.
        The HitTracker then debounces releases before audio edge-triggers.
        """
        warped = self.warp(depth_array)
        active_idx = detect_hits_blobs(warped, self.label_map, len(self.keyboard),
                                       self.floor_depth, self.threshold,
                                       min_hit_pixels=self.min_hit_pixels,
                                       max_press_height=self.max_press_height,
                                       sticky=self.tracker.held)
        active_idx = self.tracker.update(active_idx)
        active = {self.key_names[i] for i in active_idx}
        self.audio.update(active)
        return warped, active

    def stop(self):
        """Request a graceful shutdown (used by the SIGTERM handler).

        One-way flag: a SIGTERM that arrives before run() reaches its loop must
        not be overwritten by ``self._running = True``.
        """
        self._stop_requested = True
        self._running = False

    def run(self):
        headless = not os.environ.get('DISPLAY')  # unset OR empty -> headless
        log.info("--- FLOOR PIANO ---  floor=%dmm  trigger<%dmm  mode=%s",
                 self.floor_depth, self.floor_depth - self.threshold,
                 "headless" if headless else "GUI")
        if not headless:
            # DISPLAY can be set while X is actually dead (ssh leftovers, broken
            # session) — probe once and fall back to headless instead of crashing.
            try:
                cv2.namedWindow("Floor Piano - 3D View")
                cv2.waitKey(1)
                log.info("Press 'q' to quit, 'r' to re-level the floor.")
            except cv2.error as e:
                log.warning("DISPLAY is set but unusable (%s) — running headless.", e)
                headless = True

        self.camera.start()
        self._running = not self._stop_requested
        last_fps_log = time.time()
        frames_since_log = 0
        stalled_reads = 0
        corners_checked = False
        try:
            while self._running:
                depth = self.camera.read_depth()
                if depth is None:
                    stalled_reads += 1
                    if stalled_reads >= constants.CAMERA_STALL_LIMIT:
                        raise DepthCameraError(
                            f"no depth frames after {stalled_reads} consecutive reads "
                            "— camera lost or stream ended?"
                        )
                    continue
                stalled_reads = 0
                if not corners_checked:
                    self._check_corners_fit(depth)
                    corners_checked = True

                warped, active = self.process_frame(depth)
                frames_since_log += 1

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
                    fps = frames_since_log / max(now - last_fps_log, 1e-6)
                    log.info("FPS %.1f | floor %dmm | active %s", fps, self.floor_depth, sorted(active))
                    last_fps_log = now
                    frames_since_log = 0
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

    def shutdown(self):
        log.info("Shutting down...")
        self.camera.stop()
        try:
            self.audio.close()
        except Exception:
            pass
        if os.environ.get('DISPLAY'):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # X died mid-run; nothing left to close


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
