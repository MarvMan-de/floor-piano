#!/usr/bin/env python3
"""Live colourised depth viewer for the Orbbec Gemini with a stats HUD.

A small home-grown OrbbecViewer: opens a window showing the depth map in colour
plus a live overlay (FPS, resolution, pixel format, depth scale, centre distance,
distance under the mouse cursor, valid-pixel share, min/median/max mm). No
config.json / calibration needed.

    pip install pyorbbecsdk2          # imports as 'pyorbbecsdk'
    python3 tools/view_gemini.py      # 'q' or Esc to quit
    python3 tools/view_gemini.py --near 300 --far 3000   # colour range in mm

Needs a display (run on the laptop, not headless). For a terminal-only numeric
read-out use tools/probe_gemini.py instead. The SDK calls mirror DepthCamera, so
what you see here is what the real pipeline gets.
"""

import argparse
import sys
import time

import cv2
import numpy as np

_cursor = [0, 0]  # last mouse position, filled by the callback


def _on_mouse(event, x, y, flags, param):
    _cursor[0], _cursor[1] = x, y


def _draw_hud(bgr, lines):
    """Translucent panel + text, top-left, readable over any colour."""
    pad, lh = 8, 22
    h_box = pad * 2 + lh * len(lines)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (430, h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0, bgr)
    for i, t in enumerate(lines):
        cv2.putText(bgr, t, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser(description="Live Gemini depth viewer with HUD.")
    p.add_argument("--near", type=float, default=200.0, help="near colour bound (mm)")
    p.add_argument("--far", type=float, default=4000.0, help="far colour bound (mm)")
    args = p.parse_args(argv)

    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    except ImportError:
        print("FAIL: pyorbbecsdk not installed. Run 'pip install pyorbbecsdk2'.", file=sys.stderr)
        return 1

    pipeline = Pipeline()
    config = Config()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    profile = None
    try:
        profile = profile_list.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
    except Exception:
        pass
    if profile is None:
        profile = profile_list.get_default_video_stream_profile()
    if profile is None:
        print("FAIL: no depth stream profile available.", file=sys.stderr)
        return 1
    config.enable_stream(profile)
    try:
        fmt = str(profile.get_format())
    except Exception:
        fmt = "?"

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"FAIL: could not start depth stream: {e}", file=sys.stderr)
        return 1

    win = "Gemini depth — q/Esc to quit"
    try:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, _on_mouse)
    except cv2.error as e:
        print(f"FAIL: no display available ({e}). Use tools/probe_gemini.py headless.",
              file=sys.stderr)
        pipeline.stop()
        return 1

    span = max(args.far - args.near, 1.0)
    scale = None
    frames_in_window, window_start, fps = 0, time.monotonic(), 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                continue
            depth = frames.get_depth_frame()
            if depth is None:
                continue
            frames_in_window += 1
            now = time.monotonic()
            if now - window_start >= 0.5:
                fps = frames_in_window / (now - window_start)
                frames_in_window, window_start = 0, now

            w, h = depth.get_width(), depth.get_height()
            if scale is None:
                try:
                    scale = float(depth.get_depth_scale())
                except Exception:
                    scale = 1.0
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
            if raw.size != w * h:
                continue
            mm = raw.reshape(h, w).astype(np.float32) * scale

            # Colourise: map [near, far] mm onto a colour map; invalid (0) -> black.
            norm = np.clip((mm - args.near) / span, 0.0, 1.0)
            bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            bgr[mm == 0] = (0, 0, 0)

            valid = mm[mm > 0]
            cx, cy = w // 2, h // 2
            cv2.drawMarker(bgr, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
            mxv = max(0, min(_cursor[0], w - 1))
            myv = max(0, min(_cursor[1], h - 1))
            cur_mm = float(mm[myv, mxv])
            cv2.circle(bgr, (mxv, myv), 4, (255, 255, 255), 1)

            lines = [
                f"{w}x{h}  {fmt}  scale={scale:.3f}  fps={fps:4.1f}",
                f"centre: {mm[cy, cx]:6.0f} mm     cursor: {cur_mm:6.0f} mm",
            ]
            if valid.size:
                lines.append(f"valid {100 * valid.size / mm.size:3.0f}%   "
                             f"min {valid.min():.0f}  med {np.median(valid):.0f}  max {valid.max():.0f} mm")
            else:
                lines.append("NO valid depth (all zero) — too close/far or occluded?")
            lines.append(f"colour range: {args.near:.0f}..{args.far:.0f} mm")
            _draw_hud(bgr, lines)

            cv2.imshow(win, bgr)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
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
