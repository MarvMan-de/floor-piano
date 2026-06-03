"""Single place that talks to the Orbbec depth sensor via pyorbbecsdk.

Importing this module does NOT require pyorbbecsdk — the SDK is imported lazily
inside ``start()``, so the rest of the codebase (and the test suite) can import
freely on a machine without the camera SDK. Failures are raised as
``DepthCameraError`` instead of calling ``sys.exit()``, so callers decide how to
react.

This centralises what used to be duplicated between main.py and calibrate.py and
adds two safety nets discovered in the code review:
  * it requests an explicit 16-bit (Y16) depth profile, falling back to the
    device default if that is unavailable (so behaviour is never worse than before);
  * it validates the frame size before reshaping, returning None on a mismatch
    instead of crashing the loop.
"""

import logging
import os

import numpy as np

import constants
from detection import decode_depth

log = logging.getLogger(__name__)


class DepthCameraError(RuntimeError):
    """Raised when the depth camera cannot be opened or started."""


class DepthCamera:
    def __init__(self, timeout_ms=100):
        self.timeout_ms = timeout_ms
        self._pipeline = None
        self._scale = None  # raw-unit -> millimetre factor, read from the first frame

    def start(self):
        """Open and start the depth stream. Raises DepthCameraError on failure."""
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType
        except ImportError as e:
            raise DepthCameraError(
                "pyorbbecsdk is not installed. On the Pi (arm64) a plain "
                "'pip install pyorbbecsdk' is usually not enough — build/install the "
                "OrbbecSDK Python bindings and the udev rules. See docs/SETUP.md."
            ) from e

        try:
            self._pipeline = Pipeline()
            config = Config()
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            profile = self._select_depth_profile(profile_list)
            if profile is None:
                raise DepthCameraError(
                    "no depth stream profile available — is this a Legacy Astra Pro "
                    "(OpenNI2/UVC)? See docs/SETUP.md blocker #1."
                )
            config.enable_stream(profile)
            self._pipeline.start(config)
        except Exception as e:
            self._pipeline = None
            raise DepthCameraError(f"failed to start Orbbec depth stream: {e}") from e

        log.info("Orbbec depth stream started.")
        return self

    @staticmethod
    def _select_depth_profile(profile_list):
        """Prefer a Y16 (16-bit) depth profile; fall back to the device default."""
        try:
            from pyorbbecsdk import OBFormat
            # width=0, height=0, fps=0 mean "any" in pyorbbecsdk.
            profile = profile_list.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
            if profile is not None:
                return profile
        except Exception:
            log.warning("Y16 depth profile unavailable; using device default profile.")
        return profile_list.get_default_video_stream_profile()

    def read_depth(self):
        """Return the latest depth frame as an HxW uint16 numpy array, or None.

        None means "no frame this cycle" or "unexpected frame format" — the caller
        should simply continue to the next iteration.
        """
        if self._pipeline is None:
            raise DepthCameraError("camera not started; call start() first")

        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None

        h = depth_frame.get_height()
        w = depth_frame.get_width()
        depth = decode_depth(depth_frame.get_data(), h, w)
        if depth is None:
            log.warning("Unexpected depth frame size (not 16-bit %dx%d); skipping frame.", w, h)
            return None

        # Read the device's depth scale once and convert raw units to millimetres,
        # so the mm-based trigger threshold is correct on any device. Astra is
        # usually 1.0 (already mm); only convert when it differs.
        if self._scale is None:
            try:
                self._scale = float(depth_frame.get_depth_scale())
            except Exception:
                self._scale = 1.0
            log.info("Depth scale: %.4f (raw unit -> mm)", self._scale)
        if abs(self._scale - 1.0) > 1e-3:
            depth = (depth.astype(np.float32) * self._scale)
        return depth

    def stop(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class VideoDepthCamera:
    """A DepthCamera-compatible source that replays an MP4 (or any video file).

    Drop-in replacement for DepthCamera so the full warp -> detect -> audio path
    can be exercised from a *recording* without an Orbbec camera attached:

        FloorPiano(config, camera=VideoDepthCamera("clip.mp4"))

    A normal video carries no real depth, so every frame is converted to
    grayscale and its brightness mapped linearly onto a millimetre depth range.
    By DEFAULT a dark object is treated as "closer" (a foot dipping toward the
    camera) and bright pixels as the floor — tune ``floor_depth`` / ``near_depth``
    / ``invert`` until your clip actually triggers the keys.

    cv2 is imported lazily inside ``start()`` so importing this module stays
    cheap and dependency-free, consistent with the rest of the file.
    """

    def __init__(self, path, floor_depth=None, near_depth=None,
                 invert=False, loop=False):
        self.path = path
        self.floor_depth = int(floor_depth) if floor_depth is not None \
            else constants.DEFAULT_FLOOR_DEPTH
        # Map the brightest/darkest pixel comfortably above the trigger plane so a
        # full-intensity "foot" fires by default (floor - 200mm, like sweep_frames).
        self.near_depth = int(near_depth) if near_depth is not None \
            else max(1, self.floor_depth - 200)
        self.invert = bool(invert)
        self.loop = bool(loop)
        self._cap = None
        self.width = None
        self.height = None

    def start(self):
        import cv2
        if not os.path.exists(self.path):
            raise DepthCameraError(f"video file not found: {self.path}")
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            raise DepthCameraError(
                f"could not open video '{self.path}' — is it a readable mp4? "
                "OpenCV needs the right codecs (try `pip install opencv-contrib-python`)."
            )
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Video source opened: %s (%dx%d)", self.path, self.width, self.height)
        return self

    def read_depth(self):
        """Return the next frame as an HxW uint16 depth-like array, or None at EOF."""
        import cv2
        if self._cap is None:
            raise DepthCameraError("video not started; call start() first")

        ok, frame = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.invert:
            gray = 255.0 - gray
        # dark (0) -> near_depth (close, triggers); bright (255) -> floor_depth.
        span = self.floor_depth - self.near_depth
        depth = self.near_depth + (gray / 255.0) * span
        return depth.astype(np.uint16)

    def stop(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class MockDepthCamera:
    """A DepthCamera-compatible fake that serves preset frames — no hardware/SDK.

    Drop-in replacement for DepthCamera so main.FloorPiano can be run on the Pi
    (or anywhere) without an Orbbec camera attached:

        FloorPiano(config, camera=MockDepthCamera(frames))
    """

    def __init__(self, frames, loop=False):
        self._frames = list(frames)
        self._loop = loop
        self._i = 0

    def start(self):
        return self

    def read_depth(self):
        if self._i >= len(self._frames):
            if not self._loop:
                return None
            self._i = 0
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def stop(self):
        pass
