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

import numpy as np

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
