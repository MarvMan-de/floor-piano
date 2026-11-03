from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np


def _letterbox(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Fit img into a tw x th frame keeping aspect ratio (black bars, no distortion).

    Works for both BGR (HxWx3) and single-channel depth (HxW). Depth is resized
    with nearest-neighbour so distances / 'no reading' holes are never blended.
    Colour and its aligned depth get the SAME scale + offset, so they stay
    pixel-registered after letterboxing.
    """
    h, w = img.shape[:2]
    if w == tw and h == th:
        return img
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    shape = (th, tw) if img.ndim == 2 else (th, tw, img.shape[2])
    canvas = np.zeros(shape, dtype=img.dtype)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _color_to_bgr(color_frame) -> Optional[np.ndarray]:
    """Decode an Orbbec colour frame into a BGR image, or None on unknown format."""
    w, h = color_frame.get_width(), color_frame.get_height()
    name = str(color_frame.get_format()).upper()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    try:
        if "MJPG" in name or "MJPEG" in name or "JPEG" in name:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if "RGB" in name:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if "BGR" in name:
            return data.reshape(h, w, 3)
        if "YUYV" in name or "YUY2" in name:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        if "NV12" in name:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
        if "I420" in name:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    except Exception:
        return None
    return None


class MediaSource:
    """Thread-safe media source: live camera or video-file playback.

    A single daemon thread reads frames from either a cv2.VideoCapture device
    (live camera) or a video file and exposes the latest frame via read_frame().
    Source, seek, and playback-state changes are applied through pending-state
    objects that the capture thread picks up on its next iteration, avoiding
    concurrent cap.read() / cap.set() races.

    trigger_queue receives {tile_id, t} dicts emitted by the on_frame_cb
    callback set by the server after tile config is loaded.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480,
                 source_type: str = "orbbec") -> None:
        self._lock = threading.Lock()

        # Current source
        self._source_type: str = source_type   # "orbbec" | "camera" | "video"
        self._camera_index: int = camera_index
        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Orbbec source (pyorbbecsdk): the Gemini's COLOUR stream as a BGR frame,
        # resized to (width, height). Colour — not depth — so the printed piano
        # keys are visible for drawing tiles.
        self._ob: Any = None
        # D2C alignment: when detect_enabled, also grab depth aligned to colour
        # and letterbox it identically, so depth[y,x] matches the colour tiles.
        self._align: Any = None
        # Sensor-recommended post-processing filters (noise removal, spatial/
        # temporal smoothing …) — the same chain OrbbecViewer applies, which is
        # why its depth looks much cleaner than raw frames.
        self._depth_filters: list = []
        self._empty_reads: int = 0   # consecutive empty reads (bag end detection)
        self._depth: Optional[np.ndarray] = None   # latest aligned depth (uint16 mm, WxH)
        # every aligned depth frame since the last drain — the detector must
        # see the stream GAPLESSLY (a toe's first contact is a single frame)
        self._depth_q: list = []
        self.detect_enabled: bool = False

        # Frame dimensions
        self.width = width
        self.height = height

        # Latest decoded frame (under _lock)
        self._frame: Optional[np.ndarray] = None

        # Video playback state (under _lock)
        self._playing: bool = True
        self._speed: float = 1.0
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._current_frame_idx: int = 0

        # Pending changes consumed by the capture thread (under _lock)
        self._pending_source: Optional[Dict[str, Any]] = None
        self._pending_seek: Optional[int] = None

        # Trigger queue — filled by on_frame_cb, drained by /api/triggers
        self.trigger_queue: queue.Queue = queue.Queue(maxsize=500)

        # Optional per-frame callback (set by server for tile detection)
        self.on_frame_cb: Optional[Callable[[np.ndarray], None]] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> "MediaSource":
        if self._source_type == "orbbec":
            self._ob = self._open_orbbec()
            if self._ob is None:  # no depth camera / SDK -> fall back to a webcam
                self._source_type = "camera"
                self._cap = self._open_camera(self._camera_index)
        else:
            self._cap = self._open_camera(self._camera_index)
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="media-capture"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ob is not None:
            try:
                self._ob.stop()
            except Exception:
                pass
            self._ob = None
        self._align = None
        self._depth = None
        self._depth_filters = []

    def read_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest BGR frame, or None if not yet available."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def switch_source(self, source_type: str, path: Optional[str] = None,
                      camera_index: Optional[int] = None) -> None:
        """Request a source change. Applied by the capture thread on its next tick."""
        with self._lock:
            self._pending_source = {
                "type": source_type,
                "path": path,
                "camera_index": camera_index if camera_index is not None else self._camera_index,
            }

    def seek(self, frame_idx: int) -> None:
        """Request a seek to frame_idx. Applied by the capture thread."""
        with self._lock:
            self._pending_seek = frame_idx

    def set_playback(self, action: str, speed: Optional[float] = None) -> None:
        with self._lock:
            if action == "play":
                self._playing = True
            elif action == "pause":
                self._playing = False
            if speed is not None:
                self._speed = float(speed)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source_type": self._source_type,
                "video_path": self._video_path,
                "playing": self._playing,
                "speed": self._speed,
                "fps": self._fps,
                "frame": self._current_frame_idx,
                "total_frames": self._total_frames,
            }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _open_camera(self, index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _open_video(self, path: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(path)

    def _open_orbbec(self, bag_path: Optional[str] = None):
        """Start a pyorbbecsdk COLOUR (+DEPTH, D2C-aligned) pipeline, or None.

        With ``bag_path`` the frames come from a recorded .bag file instead of
        the live camera (PlaybackDevice) — same streams, same alignment, so
        the whole tile/press pipeline works on recordings unchanged.
        """
        self._align = None
        try:
            from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat,
                                     AlignFilter, OBStreamType)
        except Exception as e:
            print(f"[camera_source] pyorbbecsdk not available ({e}) — no Orbbec camera")
            return None
        try:
            if bag_path:
                from pyorbbecsdk import PlaybackDevice
                pipe = Pipeline(PlaybackDevice(bag_path))
            else:
                # Pipeline() with NO device attached does not raise cleanly —
                # on Windows the SDK dies with a heap-corruption crash
                # (0xC0000374) that takes the whole server process with it
                # before uvicorn ever binds its port ("connection refused").
                # Enumerating first is crash-safe, so bail out here instead.
                from pyorbbecsdk import Context
                if Context().query_devices().get_count() == 0:
                    print("[camera_source] no Orbbec device attached — "
                          "falling back (webcam / video / bag)")
                    return None
                pipe = Pipeline()
            cfg = Config()
            plist = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            prof = None
            for f in ("MJPG", "RGB"):
                try:
                    prof = plist.get_video_stream_profile(0, 0, getattr(OBFormat, f), 0)
                    if prof:
                        break
                except Exception:
                    pass
            prof = prof or plist.get_default_video_stream_profile()
            cfg.enable_stream(prof)
            # Depth too — best effort; the colour path still works if it fails.
            try:
                dlist = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                dprof = None
                try:
                    dprof = dlist.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
                except Exception:
                    pass
                dprof = dprof or dlist.get_default_video_stream_profile()
                cfg.enable_stream(dprof)
                self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            except Exception as e:
                print(f"[camera_source] depth stream unavailable ({e}) — colour only")
            pipe.start(cfg)
            self._depth_filters = self._load_depth_filters(pipe)
            print("[camera_source] Orbbec started (Gemini RGB%s)"
                  % (" + aligned depth" if self._align else ""))
            return pipe
        except Exception as e:
            print(f"[camera_source] Orbbec start failed ({e})")
            return None

    @staticmethod
    def _load_depth_filters(pipe) -> list:
        """The depth sensor's recommended post-processing chain, with the
        quality filters switched ON (the clean OrbbecViewer look): noise
        removal (speckles / flying pixels at foot edges), edge-preserving
        spatial smoothing and temporal smoothing (depth wobble). Decimation is
        excluded: it would change the frame size and break the surface/tile
        registration."""
        # Speckle removal only. Spatial/temporal smoothing made the depth
        # PRETTIER but smeared away the small first-contact patch of a toe
        # landing on a narrow black key (verified against a recorded bag:
        # raw depth detects the press, smoothed depth mis-assigns it) — the
        # detector's own noise/stability gates do that job instead.
        WANT_ON = ("NoiseRemoval", "EdgeNoiseRemoval")
        try:
            from pyorbbecsdk import OBSensorType
            sensor = pipe.get_device().get_sensor(OBSensorType.DEPTH_SENSOR)
            filters = []
            for f in sensor.get_recommended_filters():
                name = f.get_name()
                if "Decimation" in name:
                    continue
                if "Temporal" in name or "SpatialAdvanced" in name:
                    continue  # smoothing, see WANT_ON note
                if any(w in name for w in WANT_ON) and not f.is_enabled():
                    try:
                        f.enable(True)
                    except Exception:
                        pass
                if f.is_enabled():
                    filters.append(f)
            if filters:
                print("[camera_source] depth filters: "
                      + ", ".join(f.get_name() for f in filters))
            return filters
        except Exception as e:
            print(f"[camera_source] no depth post-processing ({e})")
            return []

    def _filter_depth(self, depth_frame):
        """Run the recommended post-processing chain on one depth frame."""
        for f in self._depth_filters:
            try:
                out = f.process(depth_frame)
                if out is None:
                    continue
                if not hasattr(out, "get_width") and hasattr(out, "as_video_frame"):
                    out = out.as_video_frame()
                depth_frame = out
            except Exception:
                pass  # a failing filter must never take down the stream
        return depth_frame

    def _read_orbbec(self) -> None:
        """Grab one Gemini frame; store colour (letterboxed) and, when detection
        is on, the D2C-aligned depth letterboxed the same way."""
        if self._ob is None:
            time.sleep(0.05)
            return
        try:
            frames = self._ob.wait_for_frames(100)
        except Exception:
            frames = None
        if frames is None:
            # A bag that stopped delivering frames has reached its end — loop
            # it by reopening the playback pipeline (like the video source).
            if self._source_type == "bag" and self._video_path:
                self._empty_reads += 1
                if self._empty_reads >= 15:  # ~1.5s without frames
                    self._empty_reads = 0
                    try:
                        self._ob.stop()
                    except Exception:
                        pass
                    self._ob = self._open_orbbec(bag_path=self._video_path)
            return
        self._empty_reads = 0

        # Only pay for D2C alignment while a play/test mode needs depth.
        if self.detect_enabled and self._align is not None:
            try:
                aligned = self._align.process(frames)
            except Exception:
                aligned = None
            if aligned is not None:
                self._store_depth(aligned.get_depth_frame())
                frames = aligned  # colour comes from the aligned set too

        color = frames.get_color_frame()
        if color is None:
            return
        bgr = _color_to_bgr(color)
        if bgr is None:
            return
        # Fit into the frontend's fixed (width, height) space WITHOUT distorting:
        # aspect-preserving scale + letterbox, so proportions stay true and the
        # SVG overlay / tiles.json stay in one coordinate system.
        self._store_frame(_letterbox(bgr, self.width, self.height))

    def _store_depth(self, depth_frame) -> None:
        """Letterbox the aligned depth into (width,height) mm and store it."""
        if depth_frame is None:
            return
        depth_frame = self._filter_depth(depth_frame)
        w, h = depth_frame.get_width(), depth_frame.get_height()
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        if raw.size != w * h:
            return
        depth = _letterbox(raw.reshape(h, w), self.width, self.height).copy()
        # The Gemini marks invalid/overflow pixels as 0xFFFF (seen live as
        # "max 65535 mm"). Map those to 0 ("no reading") so they can't poison
        # the captured surface reference or the contact-band detection.
        depth[depth >= 60000] = 0
        with self._lock:
            self._depth = depth
            self._depth_q.append(depth)
            if len(self._depth_q) > 8:
                del self._depth_q[0]

    def drain_depth_aligned(self) -> "list[np.ndarray]":
        """All aligned depth frames since the last drain (oldest first).

        For the press detector: unlike read_depth_aligned() this never skips a
        frame, so single-frame events (the first touch of a landing toe)
        reach the temporal gates exactly as in an offline replay."""
        with self._lock:
            frames, self._depth_q = self._depth_q, []
        return frames

    def read_depth_aligned(self) -> Optional[np.ndarray]:
        """Latest D2C-aligned, letterboxed depth (uint16 mm) — same space as tiles."""
        with self._lock:
            return None if self._depth is None else self._depth.copy()

    def set_detect(self, enabled: bool) -> None:
        """Toggle depth alignment/capture (only paid for in play/test mode)."""
        self.detect_enabled = bool(enabled)

    def _apply_pending_source(self) -> bool:
        """Consume _pending_source if set. Returns True if a swap occurred."""
        with self._lock:
            pending = self._pending_source
            if pending is None:
                return False
            self._pending_source = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ob is not None:
            try:
                self._ob.stop()
            except Exception:
                pass
            self._ob = None

        if pending["type"] == "orbbec":
            self._ob = self._open_orbbec()
            with self._lock:
                self._source_type = "orbbec" if self._ob is not None else "camera"
                self._video_path = None
                self._playing = True
                self._fps = 30.0
                self._total_frames = 0
                self._current_frame_idx = 0
            if self._ob is None:
                self._cap = self._open_camera(self._camera_index)
            return True

        if pending["type"] == "bag":
            path = pending["path"]
            self._ob = self._open_orbbec(bag_path=path)
            with self._lock:
                self._source_type = "bag" if self._ob is not None else "camera"
                self._video_path = path if self._ob is not None else None
                self._playing = True
                self._fps = 30.0
                self._total_frames = 0
                self._current_frame_idx = 0
            if self._ob is None:
                self._cap = self._open_camera(self._camera_index)
            return True

        if pending["type"] == "camera":
            self._cap = self._open_camera(pending["camera_index"])
            with self._lock:
                self._source_type = "camera"
                self._video_path = None
                self._camera_index = pending["camera_index"]
                self._playing = True
                self._fps = 30.0
                self._total_frames = 0
                self._current_frame_idx = 0
        else:
            path = pending["path"]
            self._cap = self._open_video(path)
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            with self._lock:
                self._source_type = "video"
                self._video_path = path
                self._fps = fps
                self._total_frames = total
                self._current_frame_idx = 0
                self._playing = True
        return True

    def _apply_pending_seek(self, reset_timer: "list[float]") -> None:
        """Consume _pending_seek if set."""
        with self._lock:
            idx = self._pending_seek
            self._pending_seek = None
        if idx is not None and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            with self._lock:
                self._current_frame_idx = idx
            reset_timer[0] = 0.0  # force immediate next read

    def _store_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
        cb = self.on_frame_cb
        source = self._source_type
        # Fire per-frame detection for video (brightness) and orbbec (depth). The
        # orbbec callback reads the aligned depth via read_depth_aligned().
        if cb is not None and source in ("video", "orbbec", "bag"):
            try:
                cb(frame)
            except Exception:
                pass

    # ── Capture thread ────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        last_video_ts: list[float] = [0.0]  # mutable list so _apply_pending_seek can reset it

        while self._running:
            self._apply_pending_source()
            self._apply_pending_seek(last_video_ts)

            with self._lock:
                mode = self._source_type
                playing = self._playing
                speed = self._speed
                fps = self._fps

            if mode in ("orbbec", "bag"):
                self._read_orbbec()
                continue

            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.033)
                continue

            if mode == "camera":
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._store_frame(frame)
                else:
                    time.sleep(0.033)
                continue

            # ── Video mode ────────────────────────────────────────────
            if not playing:
                time.sleep(0.033)
                continue

            interval = 1.0 / max(fps * speed, 1.0)
            wait = interval - (time.time() - last_video_ts[0])
            if wait > 0:
                time.sleep(wait)

            ret, frame = self._cap.read()
            last_video_ts[0] = time.time()

            if ret and frame is not None:
                pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                with self._lock:
                    self._current_frame_idx = pos
                self._store_frame(frame)
            else:
                # End of file — loop back to beginning
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self._lock:
                    self._current_frame_idx = 0
                last_video_ts[0] = 0.0
