from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np


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

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self._lock = threading.Lock()

        # Current source
        self._source_type: str = "camera"   # "camera" | "video"
        self._camera_index: int = camera_index
        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None

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

    def _apply_pending_source(self) -> bool:
        """Consume _pending_source if set. Returns True if a swap occurred."""
        with self._lock:
            pending = self._pending_source
            if pending is None:
                return False
            self._pending_source = None

        if self._cap is not None:
            self._cap.release()

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
        if cb is not None and source == "video":
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

            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.033)
                continue

            with self._lock:
                mode = self._source_type
                playing = self._playing
                speed = self._speed
                fps = self._fps

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
