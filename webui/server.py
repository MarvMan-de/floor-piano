from __future__ import annotations

"""FastAPI WebUI server for the floor-piano tile configuration system.

Run:
    python3 -m webui.server --host 0.0.0.0 --port 8000 [--camera-index N]
"""

import asyncio
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

from webui.autodetect import detect_tiles
from webui.camera_source import MediaSource
from webui.tile_store import default_tiles_doc, load_tiles, save_tiles, validate_tiles_doc
from webui.video_detect import detect_tile_hits

_REPO_ROOT = Path(__file__).parent.parent
_STATIC_DIR = Path(__file__).parent / "static"
_DEMO_DIR = _REPO_ROOT / "demo"
_VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

app = FastAPI(title="Floor Piano WebUI")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Force revalidation of the UI assets.

    Starlette's StaticFiles/FileResponse don't emit a Cache-Control header, so
    browsers fall back to *heuristic* caching and can serve a stale app.js /
    style.css after an edit without ever revalidating. Marking the UI assets
    no-cache makes the browser always check for a newer version.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# In-memory tile cache — updated on POST /api/config and on startup
_tile_cache: dict[str, Any] = default_tiles_doc()
_media: Optional[MediaSource] = None
_trigger_q: queue.Queue = queue.Queue(maxsize=500)

# Optional audio player for video mode (requires pygame + sound samples)
_audio: Any = None


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
    """Return a frame callback that detects tile hits and feeds the trigger queue."""
    def cb(frame: np.ndarray) -> None:
        tiles = _tile_cache.get("tiles", [])
        if not tiles:
            return
        triggered_ids = detect_tile_hits(frame, tiles)
        now = time.time()
        for tid in triggered_ids:
            try:
                _trigger_q.put_nowait({"tile_id": tid, "t": now})
            except queue.Full:
                pass
        # Drive audio if available
        if _audio is not None and triggered_ids:
            note_map = {t["id"]: t["note"] for t in tiles}
            active = {note_map[tid] for tid in triggered_ids if tid in note_map}
            try:
                _audio.update(active)
            except Exception:
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
    # Refresh audio note list when config changes
    global _audio
    _audio = _try_init_audio()
    return JSONResponse({"ok": True})


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
    """Grab current frame and run contour-based tile detection."""
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    suggestions = detect_tiles(frame)
    return JSONResponse({"suggestions": suggestions})


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
    return JSONResponse({"cameras": [0], "videos": videos})


@app.post("/api/source")
async def set_source(request: Request) -> JSONResponse:
    """Switch the active source: {"type":"camera"} or {"type":"video","path":"demo/foo.mp4"}."""
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    body = await request.json()
    src_type = body.get("type")
    if src_type not in ("camera", "video"):
        raise HTTPException(status_code=422, detail="type must be 'camera' or 'video'")
    path = body.get("path") if src_type == "video" else None
    if src_type == "video":
        if not path:
            raise HTTPException(status_code=422, detail="path required for video source")
        full_path = _REPO_ROOT / path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")
        _media.switch_source("video", path=str(full_path))
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
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _frame_generator() -> Generator[bytes, None, None]:
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = _media.read_frame() if _media else None
        if frame is None:
            time.sleep(0.067)
            continue

        frame = frame.copy()
        tiles = _tile_cache.get("tiles", [])

        for tile in tiles:
            if not tile.get("enabled", True):
                continue
            poly = tile.get("polygon", [])
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            color = _hex_to_bgr(tile.get("color", "#4A90D9"))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            label = tile.get("label", "")
            cv2.putText(frame, label, (cx - 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield boundary + buf.tobytes() + b"\r\n"

        time.sleep(0.067)  # ~15 FPS cap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Floor Piano WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    _tile_cache["camera_index"] = args.camera_index
    uvicorn.run(app, host=args.host, port=args.port)
