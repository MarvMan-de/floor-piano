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

        # A transient SDK hiccup (USB glitch, dropped frame) must not escape as
        # a raw OBError and kill the loop — treat it like "no frame this cycle";
        # the stall watchdog in main.py escalates if it never recovers.
        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            if frames is None:
                return None
            depth_frame = frames.get_depth_frame()
        except Exception as e:
            log.warning("Depth read failed (%s) — skipping frame.", e)
            return None
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
            # Back to integer millimetres — keeps the documented uint16 contract.
            depth = (depth.astype(np.float32) * self._scale + 0.5).astype(np.uint16)
        return depth

    def stop(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class OpenNI2DepthCamera:
    """DepthCamera-compatible source for the legacy Astra Pro via OpenNI2.

    The Astra Pro's depth sensor is NOT served by pyorbbecsdk (OrbbecSDK v2):
    it is a legacy OpenNI2 device (depth over OpenNI2, RGB as a *separate* UVC
    webcam). This backend reads depth through the ``openni`` Python bindings
    (``pip install openni``), which wrap ``libOpenNI2.so`` + the Orbbec driver,
    and returns the same HxW uint16 *millimetre* frames as DepthCamera — so
    FloorPiano needs no other change. Select it at runtime:

        FLOOR_PIANO_CAMERA=openni2 python3 src/main.py

    The PIXEL_FORMAT_DEPTH_1_MM video mode makes frame values already
    millimetres (depth scale 1.0), so no per-device scaling is needed.

    NOTE: written WITHOUT the camera attached — see docs/ASTRA_PRO_PI5_SETUP.md.
    The control flow and OpenNI2 calls follow the documented bindings, but the
    resolution/fps the device actually advertises and the exact buffer layout
    can only be confirmed on real hardware (marked TODO below). Everything that
    can be tested without the device is covered in tests/test_openni2_camera.py.
    """

    def __init__(self, width=640, height=480, fps=30):
        # 640x480@30 is the Astra Pro's standard depth mode; set_video_mode is
        # best-effort below, so a device that only offers its default mode still
        # works. TODO(hardware): confirm the advertised mode list.
        self.width = width
        self.height = height
        self.fps = fps
        self._dev = None
        self._stream = None

    def start(self):
        """Open and start the OpenNI2 depth stream. Raises DepthCameraError."""
        try:
            from openni import openni2, _openni2 as c_api
        except ImportError as e:
            raise DepthCameraError(
                "openni bindings not installed. Run 'pip install openni' AND install "
                "the Orbbec OpenNI2 SDK redist (point OPENNI2_REDIST at it). "
                "See docs/ASTRA_PRO_PI5_SETUP.md."
            ) from e

        # OPENNI2_REDIST points initialize() at libOpenNI2.so + the Orbbec driver.
        # Passing it explicitly avoids 'NiInitialize failed' when the env file
        # isn't sourced (e.g. under systemd).
        redist = os.environ.get("OPENNI2_REDIST") or os.environ.get("OPENNI2_REDIST64")
        try:
            openni2.initialize(redist) if redist else openni2.initialize()
            self._dev = openni2.Device.open_any()
            self._stream = self._dev.create_depth_stream()
            try:
                self._stream.set_video_mode(c_api.OniVideoMode(
                    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                    resolutionX=self.width, resolutionY=self.height, fps=self.fps))
            except Exception as e:
                log.warning("Could not set %dx%d@%dfps depth mode (%s) — using device "
                            "default.", self.width, self.height, self.fps, e)
            self._stream.start()
        except DepthCameraError:
            raise
        except Exception as e:
            self.stop()
            raise DepthCameraError(
                f"failed to start Astra Pro depth via OpenNI2: {e}. If this is a "
                "'USB endpoint not found' error, see the troubleshooting section in "
                "docs/ASTRA_PRO_PI5_SETUP.md (known ARM64 SDK issue)."
            ) from e

        log.info("Astra Pro depth stream started via OpenNI2.")
        return self

    def read_depth(self):
        """Return the latest depth frame as an HxW uint16 mm array, or None."""
        if self._stream is None:
            raise DepthCameraError("camera not started; call start() first")

        # A transient read hiccup must not kill the loop — treat it as "no frame
        # this cycle"; main.py's stall watchdog escalates if it never recovers.
        try:
            frame = self._stream.read_frame()
        except Exception as e:
            log.warning("OpenNI2 depth read failed (%s) — skipping frame.", e)
            return None
        if frame is None:
            return None

        # primesense/openni bindings expose .height/.width as attributes; some
        # builds use getters. TODO(hardware): confirm which on the real device.
        h = frame.height if hasattr(frame, "height") else frame.get_height()
        w = frame.width if hasattr(frame, "width") else frame.get_width()
        # get_buffer_as_uint16() exposes the frame via the buffer protocol;
        # decode_depth guards the byte count and returns a contiguous, owned
        # uint16 copy (the SDK recycles its buffer on the next read).
        depth = decode_depth(frame.get_buffer_as_uint16(), h, w)
        if depth is None:
            log.warning("Unexpected OpenNI2 depth frame size (%dx%d); skipping frame.", w, h)
        return depth

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        self._dev = None
        try:
            from openni import openni2
            openni2.unload()
        except Exception:
            pass


class VideoDepthCamera:
    """A DepthCamera-compatible source that replays an MP4 (or any video file).

    Drop-in replacement for DepthCamera so the full warp -> detect -> audio path
    can be exercised from a *recording* without an Orbbec camera attached:

        FloorPiano(config, camera=VideoDepthCamera("clip.mp4"))

    Two ways to turn an ordinary (depth-less) clip into depth-like frames:

    * brightness mode (default): each frame is converted to grayscale and its
      brightness mapped linearly onto a millimetre range. A dark object is
      treated as "closer" (a foot) and bright pixels as the floor — tune
      ``floor_depth`` / ``near_depth`` / ``invert``. Simple, but it cannot tell a
      real foot from a dark *painted* key.
    * motion mode (``motion=True``): pixels that differ from a fixed MEDIAN
      background (sampled across the whole clip in ``start()``) are the foot
      and become "close"; everything else is floor. A fixed background — unlike
      a running MOG2 subtractor — never absorbs a foot that stands still on a
      key (which would release + retrigger the note) and leaves no "ghost"
      foreground behind when the foot moves away.

    After every ``read_depth()`` the (rotated) colour frame is kept in
    ``last_bgr`` and the median background in ``background`` — demo_video.py
    uses them for the visual overlay and the mat auto-calibration.

    cv2 is imported lazily inside ``start()`` so importing this module stays
    cheap and dependency-free, consistent with the rest of the file.
    """

    BG_SAMPLES = 25  # frames sampled across the clip for the median background

    def __init__(self, path, floor_depth=None, near_depth=None,
                 invert=False, loop=False, rotate=0, motion=False,
                 motion_min_area=0.003, motion_threshold=35,
                 trigger_threshold=constants.DEFAULT_TRIGGER_THRESHOLD):
        self.path = path
        self.floor_depth = int(floor_depth) if floor_depth is not None \
            else constants.DEFAULT_FLOOR_DEPTH
        trigger_threshold = int(trigger_threshold)
        if near_depth is not None:
            self.near_depth = int(near_depth)
            if self.near_depth < 1:
                raise ValueError(
                    f"near_depth must be >= 1mm (0 is the 'no reading' sentinel), "
                    f"got {self.near_depth}"
                )
        elif motion:
            # Binary mode: put the "foot" comfortably above the trigger plane.
            self.near_depth = max(1, self.floor_depth - 200)
        else:
            # Brightness mode maps gray 0..255 linearly onto near..floor, so the
            # trigger plane sits at gray = 255 * (1 - threshold/span). Pick the
            # span so only the darkest quarter (gray < 64) counts as a foot —
            # with the old floor-200 default the cutoff was gray < 191 and ~3/4
            # of a normal frame "triggered".
            span = round(trigger_threshold * 255.0 / (255 - 64))
            self.near_depth = max(1, self.floor_depth - span)
        if self.near_depth >= self.floor_depth - trigger_threshold:
            raise ValueError(
                f"near_depth ({self.near_depth}mm) must be below the trigger plane "
                f"(floor {self.floor_depth} - threshold {trigger_threshold} = "
                f"{self.floor_depth - trigger_threshold}mm), or nothing can ever trigger"
            )
        self.invert = bool(invert)
        self.loop = bool(loop)
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be one of 0, 90, 180, 270")
        self.rotate = rotate
        self.motion = bool(motion)
        self.motion_min_area = float(motion_min_area)  # min blob area, fraction of frame
        self.motion_threshold = int(motion_threshold)  # min per-channel diff vs background
        self._cap = None
        self.width = None
        self.height = None
        self.fps = None
        self.background = None  # median BGR background (set in start(), seekable files)
        self.last_bgr = None    # the rotated colour frame behind the last read_depth()

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
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # A 90/270 rotation swaps the reported frame dimensions.
        self.width, self.height = (h, w) if self.rotate in (90, 270) else (w, h)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._compute_background(cv2)
        if self.motion and self.background is None:
            raise DepthCameraError(
                "motion mode needs a seekable video file to build the median "
                f"background, but '{self.path}' reports no frame count."
            )
        log.info("Video source opened: %s (%dx%d @ %.1f fps, rotate=%d, mode=%s)",
                 self.path, self.width, self.height, self.fps, self.rotate,
                 "motion" if self.motion else "brightness")
        return self

    def _compute_background(self, cv2):
        """Median over frames sampled across the clip = the scene without the foot."""
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return
        samples = []
        for fidx in np.linspace(0, n - 1, min(self.BG_SAMPLES, n)).astype(int):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = self._cap.read()
            if ok:
                samples.append(self._rotated(frame, cv2))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if samples:
            self.background = np.median(np.stack(samples), axis=0).astype(np.uint8)

    def _rotated(self, frame, cv2):
        if not self.rotate:
            return frame
        rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate]
        return cv2.rotate(frame, rot)

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

        frame = self._rotated(frame, cv2)
        self.last_bgr = frame

        if self.motion:
            return self._motion_depth(frame, cv2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.invert:
            gray = 255.0 - gray
        # dark (0) -> near_depth (close, triggers); bright (255) -> floor_depth.
        span = self.floor_depth - self.near_depth
        depth = self.near_depth + (gray / 255.0) * span
        return depth.astype(np.uint16)

    def _motion_depth(self, frame, cv2):
        """Pixels differing from the median background -> near_depth; rest -> floor.

        Filtering chain to isolate a clean foot blob from a noisy RGB clip:
          1. per-channel absdiff vs the fixed background, max over channels
             (a coloured shoe on the white mat stands out in SOME channel);
          2. threshold -> raw foreground mask;
          3. open (remove specks) then close (fill the foot solid);
          4. keep only connected components above ``motion_min_area`` of the
             frame, dropping stray noise blobs that aren't a foot.
        """
        diff = cv2.absdiff(frame, self.background).max(axis=2)
        fg = (diff > self.motion_threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep only blobs big enough to be a foot (area as a fraction of the frame).
        min_area = self.motion_min_area * frame.shape[0] * frame.shape[1]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        keep = np.zeros(fg.shape, dtype=bool)
        for i in range(1, n):  # 0 is background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep |= labels == i

        depth = np.full(frame.shape[:2], self.floor_depth, dtype=np.uint16)
        depth[keep] = self.near_depth
        return depth

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


def make_depth_camera():
    """Return the depth source selected by the FLOOR_PIANO_CAMERA env var.

    Default 'orbbec' (pyorbbecsdk — Astra 2 / Femto / Gemini). Set
    FLOOR_PIANO_CAMERA=openni2 for the legacy Astra Pro, which pyorbbecsdk does
    NOT serve (see docs/ASTRA_PRO_PI5_SETUP.md). Keeps the backend choice out of
    main.py so swapping cameras is one env var, no code change.
    """
    backend = os.environ.get("FLOOR_PIANO_CAMERA", "orbbec").strip().lower()
    if backend in ("openni2", "openni", "astra", "astrapro"):
        return OpenNI2DepthCamera()
    if backend not in ("orbbec", "", "pyorbbecsdk", "default"):
        log.warning("Unknown FLOOR_PIANO_CAMERA=%r — falling back to 'orbbec'.", backend)
    return DepthCamera()
