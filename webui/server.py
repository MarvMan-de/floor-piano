from __future__ import annotations

"""FastAPI WebUI server for the floor-piano tile configuration system.

Run:
    python3 -m webui.server --host 0.0.0.0 --port 8000 [--camera-index N]
"""

import asyncio
import logging
import queue
import sys
import time
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webui.autodetect import detect_corners, detect_tiles, tiles_from_corners
from webui.camera_source import MediaSource
from webui.depth_detect import (DepthHitTracker, build_tile_label_map,
                               detect_tile_hits_depth)
from webui.tile_store import default_tiles_doc, load_tiles, save_tiles, validate_tiles_doc
from webui.video_detect import detect_tile_hits

_REPO_ROOT = Path(__file__).parent.parent
_STATIC_DIR = Path(__file__).parent / "static"
_SOUNDS_DIR = _REPO_ROOT / "src" / "sounds"
_DEMO_DIR = _REPO_ROOT / "demo"
_VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

app = FastAPI(title="Floor Piano WebUI")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
# Serve the note samples so the UI can play a tile's tone on click (browser-side).
if _SOUNDS_DIR.is_dir():
    app.mount("/sounds", StaticFiles(directory=str(_SOUNDS_DIR)), name="sounds")


class _NoCacheStatic:
    """Pure-ASGI middleware: mark the UI assets no-cache so the browser revalidates.

    Deliberately NOT a BaseHTTPMiddleware: that kind iterates the response body,
    which buffers/breaks the infinite /video_feed MJPEG stream and floods the log
    with CancelledError tracebacks on shutdown. This only rewrites the
    response-start headers for "/" and "/static/*" and leaves streaming responses
    completely untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path == "/" or path.startswith("/static/")):
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = [h for h in message.get("headers", [])
                           if h[0].lower() != b"cache-control"]
                headers.append((b"cache-control", b"no-cache, must-revalidate"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(_NoCacheStatic)

# In-memory tile cache — updated on POST /api/config and on startup
_tile_cache: dict[str, Any] = default_tiles_doc()
_media: Optional[MediaSource] = None
_trigger_q: queue.Queue = queue.Queue(maxsize=500)

# Optional audio player for video mode (requires pygame + sound samples)
_audio: Any = None

# Play/test mode: depth-based finger-press detection against a captured surface.
_play_mode: bool = False
_surface_mm: Optional[np.ndarray] = None   # per-pixel reference depth of the empty surface
_label_map: Optional[np.ndarray] = None    # tile polygons painted into a label map
_label_ids: list = []                      # tile ids indexed by _label_map values
_depth_tracker: Optional[DepthHitTracker] = None   # edge-trigger + release debounce


def _rebuild_label_map() -> None:
    """(Re)build the tile label map from the current config (call on tile changes)."""
    global _label_map, _label_ids
    tiles = _tile_cache.get("tiles", [])
    w = int(_tile_cache.get("frame_width", 640))
    h = int(_tile_cache.get("frame_height", 480))
    _label_map, _label_ids = build_tile_label_map(tiles, w, h)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    global _media, _tile_cache, _audio
    _tile_cache = load_tiles()
    cfg = _tile_cache
    _media = MediaSource(
        camera_index=cfg.get("camera_index", 0),
        width=cfg.get("frame_width", 640),
        height=cfg.get("frame_height", 480),
    )
    _media.on_frame_cb = _make_detection_cb()
    _media.start()
    _audio = _try_init_audio()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _media is not None:
        _media.stop()
    if _audio is not None:
        try:
            _audio.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio (optional — requires pygame + sound samples in src/sounds/)
# ---------------------------------------------------------------------------

def _try_init_audio() -> Any:
    """Try to import PianoAudio from src/audio.py. Returns None if unavailable."""
    src_dir = str(_REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from audio import PianoAudio  # type: ignore[import]
        note_names = [t["note"] for t in _tile_cache.get("tiles", []) if "note" in t]
        if not note_names:
            return None
        return PianoAudio(keys=note_names)
    except Exception:
        return None


def _make_detection_cb():
    """Per-frame detection → trigger queue (browser plays the note on poll).

    Only runs in play mode. Orbbec: depth background-subtraction against the
    captured surface (finger closer than surface). Video: brightness fallback.
    A DepthHitTracker edge-triggers so a held press fires its note only once.
    """
    def cb(frame: np.ndarray) -> None:
        if not _play_mode or _depth_tracker is None:
            return
        tiles = _tile_cache.get("tiles", [])
        if not tiles:
            return
        src = _media._source_type if _media else None
        if src == "orbbec":
            if _surface_mm is None or _label_map is None:
                return
            depth = _media.read_depth_aligned()
            if depth is None or depth.shape != _surface_mm.shape:
                return
            hits = detect_tile_hits_depth(depth, _surface_mm, _label_map, _label_ids)
        elif src == "video":
            hits = set(detect_tile_hits(frame, tiles))
        else:
            return
        fired = _depth_tracker.update(hits)   # rising edges only
        now = time.time()
        for tid in fired:
            try:
                _trigger_q.put_nowait({"tile_id": tid, "t": now})
            except queue.Full:
                pass
    return cb


# ---------------------------------------------------------------------------
# Static SPA
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(_tile_cache)


@app.post("/api/config")
async def post_config(request: Request) -> JSONResponse:
    global _tile_cache
    body = await request.json()
    try:
        validate_tiles_doc(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    save_tiles(body)
    _tile_cache = body
    _rebuild_label_map()  # tiles changed -> refresh the depth-detection label map
    # Refresh audio note list when config changes
    global _audio
    _audio = _try_init_audio()
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Play / test mode: depth-based finger-press detection
# ---------------------------------------------------------------------------

@app.post("/api/play")
async def set_play(request: Request) -> JSONResponse:
    """Enter/leave play mode. {"enabled": true|false}. Enabling turns on depth
    alignment, rebuilds the tile label map and resets the hit tracker."""
    global _play_mode, _depth_tracker
    body = await request.json()
    enabled = bool(body.get("enabled", True))
    _play_mode = enabled
    if _media is not None:
        _media.set_detect(enabled)
    _depth_tracker = DepthHitTracker(release_frames=3) if enabled else None
    _rebuild_label_map()
    return JSONResponse({"ok": True, "play": _play_mode,
                         "surface_ready": _surface_mm is not None})


@app.post("/api/capture_surface")
async def capture_surface() -> JSONResponse:
    """Capture the empty surface as the per-pixel depth reference (no fingers!).

    Medians a few aligned-depth frames for noise. Requires the depth camera.
    """
    global _surface_mm
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    _media.set_detect(True)  # ensure aligned depth is flowing
    loop = asyncio.get_event_loop()
    grabbed = []
    for _ in range(12):
        d = await loop.run_in_executor(None, _media.read_depth_aligned)
        if d is not None:
            grabbed.append(d)
        await asyncio.sleep(0.05)
    if not grabbed:
        raise HTTPException(status_code=503,
                            detail="No depth frames — is the Orbbec connected and streaming?")
    _surface_mm = np.median(np.stack(grabbed), axis=0).astype(np.uint16)
    valid = float((_surface_mm > 0).mean())
    return JSONResponse({"ok": True, "valid_frac": round(valid, 2)})


# ---------------------------------------------------------------------------
# Camera / snapshot API
# ---------------------------------------------------------------------------

@app.get("/api/frame")
async def get_frame() -> StreamingResponse:
    """Return a single JPEG snapshot of the current frame."""
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Frame encoding failed")
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


@app.post("/api/autodetect")
async def autodetect() -> JSONResponse:
    """Grab current frame and auto-detect the mat -> per-key tile suggestions."""
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    suggestions = detect_tiles(frame)
    return JSONResponse({"suggestions": suggestions})


@app.post("/api/detect_corners")
async def detect_corners_api() -> JSONResponse:
    """Best-effort auto-detect of the mat's 4 corners (a starting point the user
    then drags). Returns {"corners": [[x,y]x4]} or {"corners": null}."""
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    corners = detect_corners(frame)
    return JSONResponse({"corners": corners})


@app.post("/api/generate_tiles")
async def generate_tiles(request: Request) -> JSONResponse:
    """Project the 24 piano keys onto a user-provided 4-corner mat quad.

    Body: {"corners": [[x,y]x4] in TL,TR,BR,BL, "num_white"?: int}.
    This is the reliable path: the human places/drags the corners, the keys
    follow perspective-correctly. Returns {"tiles": [{polygon,label,note}, ...]}.
    """
    body = await request.json()
    corners = body.get("corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise HTTPException(status_code=422, detail="corners must be 4 [x, y] points")
    try:
        tiles = tiles_from_corners(corners, body.get("num_white"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse({"tiles": tiles})


# ---------------------------------------------------------------------------
# Source management API
# ---------------------------------------------------------------------------

@app.get("/api/sources")
async def get_sources() -> JSONResponse:
    """List available cameras (index 0 only) and video files in demo/."""
    videos = []
    if _DEMO_DIR.exists():
        for ext in _VIDEO_EXTS:
            for p in sorted(_DEMO_DIR.glob(ext)):
                videos.append(str(p.relative_to(_REPO_ROOT)))
    return JSONResponse({"depth_camera": True, "cameras": [0], "videos": videos})


@app.post("/api/source")
async def set_source(request: Request) -> JSONResponse:
    """Switch the active source: {"type":"orbbec"|"camera"|"video","path":"demo/foo.mp4"}."""
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    body = await request.json()
    src_type = body.get("type")
    if src_type not in ("orbbec", "camera", "video"):
        raise HTTPException(status_code=422, detail="type must be 'orbbec', 'camera' or 'video'")
    if src_type == "video":
        path = body.get("path")
        if not path:
            raise HTTPException(status_code=422, detail="path required for video source")
        full_path = _REPO_ROOT / path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")
        _media.switch_source("video", path=str(full_path))
    elif src_type == "orbbec":
        _media.switch_source("orbbec")
    else:
        camera_index = body.get("camera_index", 0)
        _media.switch_source("camera", camera_index=int(camera_index))
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Playback control API (video mode only)
# ---------------------------------------------------------------------------

@app.post("/api/seek")
async def seek(request: Request) -> JSONResponse:
    """Seek video to frame index: {"frame": 1234}."""
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    body = await request.json()
    frame_idx = body.get("frame")
    if not isinstance(frame_idx, int) or frame_idx < 0:
        raise HTTPException(status_code=422, detail="frame must be a non-negative integer")
    _media.seek(frame_idx)
    return JSONResponse({"ok": True})


@app.post("/api/playback")
async def set_playback(request: Request) -> JSONResponse:
    """Control video playback: {"action":"play"|"pause","speed":1.0}."""
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    body = await request.json()
    action = body.get("action")
    if action not in ("play", "pause"):
        raise HTTPException(status_code=422, detail="action must be 'play' or 'pause'")
    speed = body.get("speed")
    if speed is not None:
        try:
            speed = float(speed)
            if speed <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="speed must be a positive number")
    _media.set_playback(action, speed)
    return JSONResponse({"ok": True})


@app.get("/api/video_status")
async def video_status() -> JSONResponse:
    """Return current playback state for timeline UI updates."""
    if _media is None:
        raise JSONResponse({"source_type": "camera", "playing": False})
    return JSONResponse(_media.get_status())


# ---------------------------------------------------------------------------
# Trigger events API
# ---------------------------------------------------------------------------

@app.get("/api/triggers")
async def get_triggers() -> JSONResponse:
    """Return tile trigger events from the last 500 ms and drain the queue."""
    cutoff = time.time() - 0.5
    events = []
    while True:
        try:
            ev = _trigger_q.get_nowait()
            if ev["t"] >= cutoff:
                events.append({"tile_id": ev["tile_id"]})
        except queue.Empty:
            break
    return JSONResponse({"triggers": events})


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------

@app.get("/video_feed")
async def video_feed(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _frame_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _frame_generator(request: Request):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        # Stop cleanly when the browser disconnects or the server shuts down,
        # instead of hanging in an await and raising CancelledError on Ctrl-C.
        if await request.is_disconnected():
            break

        frame = _media.read_frame() if _media else None
        if frame is None:
            await asyncio.sleep(0.067)
            continue

        # NOTE: tiles are drawn client-side by the SVG overlay (app.js). We must
        # NOT also burn them into the stream here — otherwise every tile appears
        # twice, and offset whenever the frame size differs from the overlay's
        # 640x480 coordinate space (this was the "doubled tiles after save" bug).
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield boundary + buf.tobytes() + b"\r\n"

        await asyncio.sleep(0.067)  # ~15 FPS cap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class _DropCancelledError(logging.Filter):
    """Silence the benign CancelledError traceback that uvicorn/asyncio log when
    the server is interrupted (Ctrl-C, incl. force-quit double Ctrl-C). The server
    still shuts down cleanly; this only stops the scary but harmless traceback.
    Walks the exception __context__/__cause__ chain because on shutdown the raised
    exception is often a KeyboardInterrupt *caused by* a CancelledError."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        seen = set()
        while exc is not None and id(exc) not in seen:
            if isinstance(exc, asyncio.CancelledError):
                return False
            seen.add(id(exc))
            exc = exc.__cause__ or exc.__context__
        return True


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Floor Piano WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    # Attach to every logger that might emit the shutdown CancelledError.
    _cancel_filter = _DropCancelledError()
    for _name in ("uvicorn.error", "uvicorn", "asyncio"):
        logging.getLogger(_name).addFilter(_cancel_filter)

    _tile_cache["camera_index"] = args.camera_index
    uvicorn.run(app, host=args.host, port=args.port)
