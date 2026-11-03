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

from webui.autodetect import (detect_corners, detect_tiles, flip_corners,
                              order_corners, tiles_from_corners)
from webui.camera_source import MediaSource
from webui.depth_detect import (DepthHitTracker, PressDetector,
                               build_tile_label_map, render_semantic)
from webui.tile_store import default_tiles_doc, load_tiles, save_tiles, validate_tiles_doc
from webui.video_detect import detect_tile_hits

_REPO_ROOT = Path(__file__).parent.parent
_STATIC_DIR = Path(__file__).parent / "static"
_SOUNDS_DIR = _REPO_ROOT / "src" / "sounds"
_DEMO_DIR = _REPO_ROOT / "demo"
_BAG_DIR = _REPO_ROOT / "video"
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
_detector: Optional[PressDetector] = None          # created by /api/capture_surface
_video_tracker: Optional[DepthHitTracker] = None   # brightness path (video files)


def _rebuild_label_map() -> None:
    """Refresh the detector's tile label map from the current config."""
    if _detector is None:
        return
    tiles = _tile_cache.get("tiles", [])
    w = int(_tile_cache.get("frame_width", 640))
    h = int(_tile_cache.get("frame_height", 480))
    _detector.set_tiles(*build_tile_label_map(tiles, w, h))


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
        if not _play_mode:
            return
        tiles = _tile_cache.get("tiles", [])
        if not tiles:
            return
        src = _media._source_type if _media else None
        if src in ("orbbec", "bag"):
            if _detector is None:
                return  # surface not captured yet
            fired = set()
            # process EVERY aligned depth frame since the last callback — a
            # skipped frame can swallow the single first-contact frame of a
            # toe landing on a key boundary
            for depth in _media.drain_depth_aligned():
                if depth.shape == _detector.surface.shape:
                    fired |= _detector.update(depth)
        elif src == "video" and _video_tracker is not None:
            fired = _video_tracker.update(set(detect_tile_hits(frame, tiles)))
        else:
            return
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
    alignment, refreshes the detector's tiles and resets its runtime state."""
    global _play_mode, _video_tracker
    body = await request.json()
    enabled = bool(body.get("enabled", True))
    _play_mode = enabled
    if _media is not None:
        _media.set_detect(enabled)
    _video_tracker = DepthHitTracker(release_frames=3) if enabled else None
    _rebuild_label_map()  # also resets the detector's streak/held state
    return JSONResponse({"ok": True, "play": _play_mode,
                         "surface_ready": _detector is not None})


async def _grab_depth_frames(count: int, loop) -> list:
    grabbed = []
    for _ in range(count):
        d = await loop.run_in_executor(None, _media.read_depth_aligned)
        if d is not None:
            grabbed.append(d)
        await asyncio.sleep(0.05)
    return grabbed


@app.post("/api/capture_surface")
async def capture_surface() -> JSONResponse:
    """Capture the empty surface as the per-pixel depth reference (no fingers!).

    Medians a few aligned-depth frames for noise, then watches a few MORE empty
    frames: pixels that already read "in contact" with nothing there are
    systematic artifacts (edges/reflections/warp mismatch) and get masked out —
    this is the fix for "the same keys keep firing on their own".
    """
    global _detector
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    _media.set_detect(True)  # ensure aligned depth is flowing
    loop = asyncio.get_event_loop()
    grabbed = await _grab_depth_frames(12, loop)
    if not grabbed:
        raise HTTPException(status_code=503,
                            detail="No depth frames — is the Orbbec connected and streaming?")
    surface = np.median(np.stack(grabbed), axis=0).astype(np.uint16)
    _detector = PressDetector(surface)
    extra = await _grab_depth_frames(8, loop)
    # All empty frames feed the artifact calibration: in-band mismatches AND
    # temporally unstable (flickering) pixels get masked out.
    artifact_frac = _detector.calibrate_artifacts(grabbed + extra)
    _rebuild_label_map()
    valid = float((surface > 0).mean())
    return JSONResponse({"ok": True, "valid_frac": round(valid, 2),
                         "artifact_frac": round(artifact_frac, 3)})


@app.post("/api/capture_feet")
async def capture_feet(request: Request) -> JSONResponse:
    """Calibrate the contact tolerance from the player's own feet.

    The player STANDS still on the captured surface; the detector measures the
    shoe-top height (contact is guaranteed while standing) and sets the band
    ceiling to it + margin. Thick sneakers get the reach they need; bare feet
    get a tighter band than the default. Requires a captured surface.

    Optional body {"ceiling_mm": N} sets the ceiling directly (25..120 mm)
    without measuring — for recordings without a standing segment."""
    if _detector is None:
        raise HTTPException(status_code=409,
                            detail="Zuerst die Oberfläche erfassen (ohne Füße)")
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    try:
        body = await request.json()
    except Exception:
        body = {}
    manual = body.get("ceiling_mm")
    if manual is not None:
        mm = int(np.clip(int(manual), 25, 120))
        _detector.contact_max = mm
        _detector._build_maps()
        return JSONResponse({"ok": True, "touch_ceiling_mm": mm, "manual": True})
    _media.set_detect(True)
    loop = asyncio.get_event_loop()
    frames = await _grab_depth_frames(10, loop)
    if not frames:
        raise HTTPException(status_code=503, detail="No depth frames")
    mm = await loop.run_in_executor(None, _detector.calibrate_touch, frames)
    if mm is None:
        raise HTTPException(
            status_code=422,
            detail="Keine Füße erkannt — mit beiden Füßen auf die Matte stellen "
                   "und still stehen")
    return JSONResponse({"ok": True, "touch_ceiling_mm": mm})


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
    # detection is CPU-heavy (hundreds of ms on the Pi) — keep the loop free
    suggestions = await loop.run_in_executor(None, detect_tiles, frame)
    return JSONResponse({"suggestions": suggestions})


@app.post("/api/detect_corners")
async def detect_corners_api() -> JSONResponse:
    """Best-effort auto-detect of the mat's 4 corners (a starting point the user
    then drags). Returns {"corners": [[x,y]x4]} or {"corners": null}."""
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")
    # detection is CPU-heavy (hundreds of ms on the Pi) — keep the loop free
    corners = await loop.run_in_executor(None, detect_corners, frame)
    return JSONResponse({"corners": corners})


@app.post("/api/generate_tiles")
async def generate_tiles(request: Request) -> JSONResponse:
    """Project the 24 piano keys onto a user-provided 4-corner mat quad.

    Body: {"corners": [[x,y]x4] in ANY order, "num_white"?: int,
           "flip_h"?: bool, "flip_v"?: bool}.
    This is the reliable path: the human places/drags the corners, the keys
    follow perspective-correctly. The corners are ordered and oriented HERE,
    against the live frame's black-key pattern (a client-side sum/diff sort
    mislabels oblique quads and collapses the projection) — the client sends
    its points as clicked. When the mat lies KEYS-DOWN (blank back up) no
    pattern is visible: orientation falls back to screen geometry and the
    user's flip_h (swap the C4 end) / flip_v (black keys along the other long
    edge) are applied on top. Returns {"tiles": [...], "corners": [[x,y]x4],
    "oriented_by": "pattern"|"geometry"}.
    """
    body = await request.json()
    corners = body.get("corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise HTTPException(status_code=422, detail="corners must be 4 [x, y] points")
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _media.read_frame if _media else lambda: None)

    def _order_and_project():
        ordered, how = order_corners(corners, frame, body.get("num_white"))
        ordered = flip_corners(ordered, bool(body.get("flip_h")),
                               bool(body.get("flip_v")))
        return ordered, how, tiles_from_corners(ordered, body.get("num_white"))

    try:
        # runs per drag tick (~120 ms debounce) and costs tens of ms — keep it
        # off the event loop so the MJPEG stream stays smooth
        ordered, how, tiles = await loop.run_in_executor(None, _order_and_project)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return JSONResponse({"tiles": tiles, "corners": ordered.tolist(),
                         "oriented_by": how})


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
    # Orbbec .bag recordings (colour + depth) — play through the FULL depth
    # pipeline: mark the tiles on the first frame, then play as with the live
    # camera.
    bags = []
    for bag_dir in (_BAG_DIR, _DEMO_DIR):
        if bag_dir.exists():
            for p in sorted(bag_dir.glob("*.bag")):
                bags.append(str(p.relative_to(_REPO_ROOT)))
    return JSONResponse({"depth_camera": True, "cameras": [0],
                         "videos": videos, "bags": bags})


@app.post("/api/source")
async def set_source(request: Request) -> JSONResponse:
    """Switch the active source: {"type":"orbbec"|"camera"|"video","path":"demo/foo.mp4"}."""
    if _media is None:
        raise HTTPException(status_code=503, detail="Media source not ready")
    body = await request.json()
    src_type = body.get("type")
    if src_type not in ("orbbec", "camera", "video", "bag"):
        raise HTTPException(status_code=422,
                            detail="type must be 'orbbec', 'camera', 'video' or 'bag'")
    if src_type in ("video", "bag"):
        path = body.get("path")
        if not path:
            raise HTTPException(status_code=422, detail="path required for video/bag source")
        full_path = _REPO_ROOT / path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")
        _media.switch_source(src_type, path=str(full_path))
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
# Depth debug stream (shown in the UI sidebar while play mode is on)
# ---------------------------------------------------------------------------

def _render_depth_view() -> Optional[np.ndarray]:
    """Colorized view of the aligned depth the detector works on, or None.

    With a captured surface: the detector's own debug rendering (band/contact/
    artifact colors). Before capture: a plain distance colormap so the stream
    is still useful for aiming the camera.
    """
    depth = _media.read_depth_aligned() if _media is not None else None
    if depth is None:
        return None
    if _detector is not None and depth.shape == _detector.surface.shape:
        return _detector.render_debug(depth)
    d = depth.astype(np.float32)
    norm = np.clip((d - 400.0) / (3000.0 - 400.0), 0.0, 1.0)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    img[depth == 0] = (25, 25, 25)
    return img


def _render_semantic_view() -> Optional[np.ndarray]:
    """Semantic scene: tiles as colored regions, the foot (hover) orange, the
    contact patch red with its centre marked, sounding tiles lit."""
    tiles = _tile_cache.get("tiles", [])
    w = int(_tile_cache.get("frame_width", 640))
    h = int(_tile_cache.get("frame_height", 480))
    hover = contact = None
    held: set = set()
    if _detector is not None and _media is not None:
        depth = _media.read_depth_aligned()
        if depth is not None and depth.shape == _detector.surface.shape:
            _band, contact, hover = _detector.live_masks(depth)
            held = set(_detector.held)
    return render_semantic(tiles, w, h, hover=hover, contact=contact, held=held)


@app.get("/depth_feed")
async def depth_feed(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _view_generator(request, _render_depth_view),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/semantic_feed")
async def semantic_feed(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _view_generator(request, _render_semantic_view),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _view_generator(request: Request, render_fn):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    loop = asyncio.get_event_loop()
    while True:
        if await request.is_disconnected():
            break
        img = await loop.run_in_executor(None, render_fn)
        if img is None:
            await asyncio.sleep(0.2)
            continue
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield boundary + buf.tobytes() + b"\r\n"
        await asyncio.sleep(0.1)  # ~10 FPS is plenty for a debug view


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
