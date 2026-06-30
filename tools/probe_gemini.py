#!/usr/bin/env python3
"""Live depth read-out for the Orbbec Gemini (pyorbbecsdk) — a realtime sanity check.

Standalone: it does NOT need config.json or calibration. It opens the depth
stream directly and prints, several times per second, the actual numbers coming
off the camera so you can confirm the format and that the data is sane BEFORE
wiring up the full piano:

    pip install pyorbbecsdk2          # imports as 'pyorbbecsdk'
    python3 tools/probe_gemini.py     # Ctrl-C to stop

Per line it shows: stream WxH + pixel format + depth scale (raw unit -> mm),
measured FPS, and for the current frame the centre distance, the share of valid
pixels, and the min / median / max distance in millimetres. A healthy Gemini
pointed at a wall/floor at ~1 m shows centre ~1000mm, valid ~>80%, and numbers
that change as you move your hand in front of it.

The SDK calls mirror src/depth_camera.py (DepthCamera) exactly, so a green run
here means the real pipeline will see the same frames.
"""

import argparse
import sys
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description="Live Gemini depth read-out.")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="auto-stop after N seconds (0 = run until Ctrl-C)")
    p.add_argument("--interval", type=float, default=0.3,
                   help="seconds between printed lines (default 0.3)")
    args = p.parse_args(argv)

    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    except ImportError:
        print("FAIL: pyorbbecsdk not installed. Run 'pip install pyorbbecsdk2' "
              "(it imports as 'pyorbbecsdk').", file=sys.stderr)
        return 1

    pipeline = Pipeline()
    config = Config()

    # Prefer an explicit 16-bit (Y16) depth profile; fall back to the device
    # default — identical to DepthCamera._select_depth_profile.
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

    # Print the negotiated profile once, so the FORMAT is on the record.
    try:
        print(f"Depth profile: {profile.get_width()}x{profile.get_height()} "
              f"@ {profile.get_fps()}fps  format={profile.get_format()}")
    except Exception as e:
        print(f"(could not read profile details: {e})")

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"FAIL: could not start the depth stream: {e}", file=sys.stderr)
        return 1
    print("Streaming — Ctrl-C to stop.\n")

    start = time.monotonic()
    last_print = 0.0
    frames_in_window = 0
    window_start = start
    scale = None

    try:
        while True:
            if args.seconds and time.monotonic() - start >= args.seconds:
                break

            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth = frames.get_depth_frame()
            if depth is None:
                continue
            frames_in_window += 1

            now = time.monotonic()
            if now - last_print < args.interval:
                continue

            w, h = depth.get_width(), depth.get_height()
            if scale is None:
                try:
                    scale = float(depth.get_depth_scale())
                except Exception:
                    scale = 1.0
            # Raw 16-bit buffer -> millimetres (mm = raw * depth_scale).
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
            if raw.size != w * h:
                print(f"  ?? unexpected buffer size {raw.size} for {w}x{h} — wrong format?")
                last_print = now
                continue
            mm = raw.reshape(h, w).astype(np.float32) * scale
            valid = mm[mm > 0]
            centre = mm[h // 2, w // 2]
            fps = frames_in_window / max(now - window_start, 1e-6)
            frames_in_window, window_start = 0, now

            if valid.size:
                line = (f"[{now - start:5.1f}s] {w}x{h} scale={scale:.3f}  fps={fps:4.1f} | "
                        f"centre={centre:6.0f}mm  valid={100 * valid.size / mm.size:3.0f}%  "
                        f"min={valid.min():5.0f} med={np.median(valid):5.0f} max={valid.max():5.0f} mm")
            else:
                line = (f"[{now - start:5.1f}s] {w}x{h} scale={scale:.3f}  fps={fps:4.1f} | "
                        f"NO valid depth pixels (all zero) — too close/far or occluded?")
            print(line)
            last_print = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
