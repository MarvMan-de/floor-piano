#!/usr/bin/env python3
"""Live viewer for the Orbbec Gemini — toggle between colourised DEPTH and RGB.

A small home-grown OrbbecViewer with a stats HUD. No config.json / calibration
needed.

    pip install pyorbbecsdk2          # imports as 'pyorbbecsdk'
    python3 tools/view_gemini.py      # 'd' depth, 'c' rgb, space toggle, q/Esc quit
    python3 tools/view_gemini.py --near 300 --far 3000   # depth colour range in mm

Depth HUD: resolution, pixel format, depth scale, FPS, centre distance, valid
pixel share, min/median/max mm. RGB view is the plain colour camera (decoded from
whatever format the device streams). Needs a display (use tools/probe_gemini.py
for a headless numeric read-out). SDK calls mirror DepthCamera.
"""

import argparse
import sys
import time

import cv2
import numpy as np

_cursor = [0, 0]  # last mouse position, filled by the callback


def _on_mouse(event, x, y, flags, param):
    _cursor[0], _cursor[1] = x, y


def color_to_bgr(color_frame):
    """Decode an Orbbec colour frame into a BGR image, or None if unknown format.

    The Gemini can stream colour as MJPG, YUYV, RGB, NV12, ... — convert by the
    reported format name so cv2 can display it. Robust to enum naming differences
    by matching on the format string.
    """
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
        if "UYVY" in name:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
        if "NV12" in name:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
        if "I420" in name:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    except Exception:
        return None
    return None


def colorize_depth(mm, near, span):
    """Millimetre depth array -> TURBO-coloured BGR; invalid (0) pixels black."""
    norm = np.clip((mm - near) / span, 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bgr[mm == 0] = (0, 0, 0)
    return bgr


def _draw_hud(bgr, lines):
    """Translucent panel + text, top-left, readable over any colour."""
    pad, lh = 8, 22
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (440, pad * 2 + lh * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0, bgr)
    for i, t in enumerate(lines):
        cv2.putText(bgr, t, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def _open_pipeline(args):
    """Start a depth (+ colour if possible) pipeline.

    Returns (pipeline, color_ok, fmt, align) where align is a D2C AlignFilter
    (depth -> colour frame) or None when there is no colour stream.
    """
    from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat,
                             AlignFilter, OBStreamType)
    pipeline = Pipeline()
    config = Config()

    # Depth: prefer Y16, fall back to default (mirrors DepthCamera).
    dlist = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    dprof = None
    try:
        dprof = dlist.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
    except Exception:
        pass
    dprof = dprof or dlist.get_default_video_stream_profile()
    if dprof is None:
        raise RuntimeError("no depth stream profile available")
    config.enable_stream(dprof)
    fmt = str(dprof.get_format())

    # Colour: best-effort — prefer easy-to-decode formats, else device default.
    color_ok = True
    try:
        clist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        cprof = None
        for f in ("MJPG", "RGB"):
            try:
                cprof = clist.get_video_stream_profile(0, 0, getattr(OBFormat, f), 0)
                if cprof:
                    break
            except Exception:
                pass
        cprof = cprof or clist.get_default_video_stream_profile()
        config.enable_stream(cprof)
    except Exception as e:
        color_ok = False
        print(f"(colour stream unavailable: {e} — depth only)")

    try:
        pipeline.start(config)
    except Exception:
        if not color_ok:
            raise
        # Combined start failed — retry depth-only so the viewer still works.
        print("(colour+depth failed to start together — running depth only)")
        config = Config()
        config.enable_stream(dprof)
        pipeline.start(config)
        color_ok = False
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if color_ok else None
    return pipeline, color_ok, fmt, align


def main(argv=None):
    p = argparse.ArgumentParser(description="Live Gemini viewer: depth <-> RGB.")
    p.add_argument("--near", type=float, default=200.0, help="depth near colour bound (mm)")
    p.add_argument("--far", type=float, default=4000.0, help="depth far colour bound (mm)")
    p.add_argument("--start", choices=("depth", "rgb"), default="depth", help="initial view")
    args = p.parse_args(argv)

    try:
        import pyorbbecsdk  # noqa: F401  (checked here for a clean message)
    except ImportError:
        print("FAIL: pyorbbecsdk not installed. Run 'pip install pyorbbecsdk2'.", file=sys.stderr)
        return 1

    try:
        pipeline, color_ok, fmt, align = _open_pipeline(args)
    except Exception as e:
        print(f"FAIL: could not start the camera: {e}", file=sys.stderr)
        return 1

    win = "Gemini — d:depth  c:rgb  o:overlay  space:cycle  q:quit"
    try:
        cv2.namedWindow(win)
        cv2.waitKey(1)
    except cv2.error as e:
        print(f"FAIL: cannot open a window ({e}). No display? Use tools/probe_gemini.py "
              "(on Wayland try: QT_QPA_PLATFORM=xcb).", file=sys.stderr)
        pipeline.stop()
        return 1
    hover = True
    try:
        cv2.setMouseCallback(win, _on_mouse)
    except cv2.error:
        hover = False

    avail = ["depth"] + (["rgb", "overlay"] if color_ok else [])
    mode = "rgb" if (args.start == "rgb" and color_ok) else "depth"
    span = max(args.far - args.near, 1.0)
    scale = None
    frames_in_window, window_start, fps = 0, time.monotonic(), 0.0
    ctrl = "[d]epth [c]rgb [o]verlay [space]cycle [q]uit" if color_ok else "[q]uit"

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is not None:
                frames_in_window += 1
                now = time.monotonic()
                if now - window_start >= 0.5:
                    fps = frames_in_window / (now - window_start)
                    frames_in_window, window_start = 0, now

                if mode == "depth":
                    depth = frames.get_depth_frame()
                    if depth is not None:
                        w, h = depth.get_width(), depth.get_height()
                        if scale is None:
                            try:
                                scale = float(depth.get_depth_scale())
                            except Exception:
                                scale = 1.0
                        raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
                        if raw.size == w * h:
                            mm = raw.reshape(h, w).astype(np.float32) * scale
                            bgr = colorize_depth(mm, args.near, span)
                            cx, cy = w // 2, h // 2
                            cv2.drawMarker(bgr, (cx, cy), (255, 255, 255),
                                           cv2.MARKER_CROSS, 16, 1)
                            valid = mm[mm > 0]
                            lines = [f"DEPTH  {w}x{h}  {fmt}  scale={scale:.3f}  fps={fps:4.1f}"]
                            if hover:
                                mx = max(0, min(_cursor[0], w - 1))
                                my = max(0, min(_cursor[1], h - 1))
                                cv2.circle(bgr, (mx, my), 4, (255, 255, 255), 1)
                                lines.append(f"centre {mm[cy, cx]:6.0f} mm   cursor {mm[my, mx]:6.0f} mm")
                            else:
                                lines.append(f"centre {mm[cy, cx]:6.0f} mm")
                            if valid.size:
                                lines.append(f"valid {100 * valid.size / mm.size:3.0f}%   "
                                             f"min {valid.min():.0f} med {np.median(valid):.0f} "
                                             f"max {valid.max():.0f} mm")
                            lines.append(ctrl)
                            _draw_hud(bgr, lines)
                            cv2.imshow(win, bgr)
                elif mode == "rgb":
                    color = frames.get_color_frame() if color_ok else None
                    img = color_to_bgr(color) if color is not None else None
                    if img is not None:
                        lines = [f"RGB  {img.shape[1]}x{img.shape[0]}  fps={fps:4.1f}", ctrl]
                        _draw_hud(img, lines)
                        cv2.imshow(win, img)

                else:  # overlay: depth (D2C-aligned) blended onto the colour image
                    aligned = align.process(frames) if align is not None else None
                    d = aligned.get_depth_frame() if aligned is not None else None
                    c = aligned.get_color_frame() if aligned is not None else None
                    cimg = color_to_bgr(c) if c is not None else None
                    if d is not None and cimg is not None:
                        w, h = d.get_width(), d.get_height()
                        if scale is None:
                            try:
                                scale = float(d.get_depth_scale())
                            except Exception:
                                scale = 1.0
                        raw = np.frombuffer(d.get_data(), dtype=np.uint16)
                        if raw.size == w * h:
                            mm = raw.reshape(h, w).astype(np.float32) * scale
                            dcol = colorize_depth(mm, args.near, span)
                            # aligned depth & colour share a resolution, but guard anyway
                            if dcol.shape[:2] != cimg.shape[:2]:
                                dcol = cv2.resize(dcol, (cimg.shape[1], cimg.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)
                                mm = cv2.resize(mm, (cimg.shape[1], cimg.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                            out_img = cimg.copy()
                            m = mm > 0  # blend depth only where it has a reading
                            out_img[m] = cv2.addWeighted(cimg, 0.45, dcol, 0.55, 0)[m]
                            lines = [f"OVERLAY (D2C)  {out_img.shape[1]}x{out_img.shape[0]}  "
                                     f"fps={fps:4.1f}", ctrl]
                            _draw_hud(out_img, lines)
                            cv2.imshow(win, out_img)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('d'):
                mode = "depth"
            elif k == ord('c') and color_ok:
                mode = "rgb"
            elif k == ord('o') and color_ok:
                mode = "overlay"
            elif k == ord(' '):
                mode = avail[(avail.index(mode) + 1) % len(avail)]
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
