"""Run the full pipeline against an MP4 (or any video file) — no camera needed.

Feeds a recorded video through the real FloorPiano (warp -> label -> detect ->
audio), turning each frame into a depth-like image (see VideoDepthCamera). Lets
you test the piano from a clip on a laptop or the Pi without an Orbbec camera.

    python3 src/demo_video.py clip.mp4 --motion
    python3 src/demo_video.py clip.mp4 --motion --out result.mp4
    python3 src/demo_video.py clip.mp4 --floor 1000 --near 700 --threshold 50

By default the key grid is AUTO-CALIBRATED from the mat itself: the mat's
corners are detected in a median background frame, the orientation resolved
from the printed black-key pattern, and the grid refined onto the painted
keys (see mat_calibration.py). The clip may be portrait or upside down — no
--rotate gymnastics needed. Use --full-frame or --corners to override.

For an ordinary RGB clip use --motion (foot = what differs from the median
background). The default brightness mode is only useful for clips that encode
depth as gray values.

Exercises everything except the camera/SDK and the RGB->depth registration.
Needs: numpy, opencv (cv2), and (unless --no-audio) pygame + the samples in
src/sounds/.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

import constants
import mat_calibration
from depth_camera import DepthCameraError, VideoDepthCamera
from detection import above_floor_mask
from main import FloorPiano

log = logging.getLogger("demo_video")


def render_frame(piano, warped, active, bgr=None, frame_no=None):
    """Return a BGR visualisation of one processed frame (the 'piano canvas').

    When the source colour frame is available it is warped onto the canvas as
    the background, so you SEE the painted mat under the key grid — any
    grid-vs-mat misalignment is immediately visible. Foot pixels are tinted
    red, the grid drawn on top, active keys filled green.
    """
    th, tw = piano.target_height, piano.target_width
    if bgr is not None:
        vis = cv2.warpPerspective(bgr, piano.M, (tw, th))
    else:
        # No colour source: show the warped depth as grayscale.
        valid = warped[warped > 0]
        if valid.size:
            lo, hi = int(valid.min()), int(valid.max())
            span = max(hi - lo, 1)
            gray = np.clip((warped.astype(np.int32) - lo) * 255 // span, 0, 255).astype(np.uint8)
            gray[warped == 0] = 0
        else:
            gray = np.zeros((th, tw), dtype=np.uint8)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask = above_floor_mask(warped, piano.floor_depth, piano.threshold)
    vis[mask] = (vis[mask] // 2 + np.array([0, 0, 127], dtype=np.uint8))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    overlay = vis.copy()
    for k in piano.keyboard:
        if k.name in active:
            cv2.rectangle(overlay, (k.x0, k.y0), (k.x1, k.y1), (0, 200, 0), -1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    for k in piano.keyboard:
        if k.kind == "white":
            color = (0, 255, 0) if k.name in active else (90, 90, 90)
            cv2.rectangle(vis, (k.x0, 0), (k.x1, th), color, 2)
            cv2.putText(vis, k.name, (k.x0 + 3, th - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for k in piano.keyboard:
        if k.kind == "black":
            color = (0, 255, 0) if k.name in active else (200, 200, 200)
            cv2.rectangle(vis, (k.x0, k.y0), (k.x1, k.y1), color, 2)

    hud = []
    if frame_no is not None:
        hud.append(f"frame {frame_no}")
    if active:
        hud.append("playing: " + " ".join(sorted(active)))
    if hud:
        cv2.putText(vis, "  ".join(hud), (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


def parse_corners(text):
    """Parse 'x,y x,y x,y x,y' (TL TR BR BL) into a 4x2 float list."""
    pts = [tuple(float(v) for v in p.split(",")) for p in text.replace(";", " ").split()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise argparse.ArgumentTypeError("need exactly 4 'x,y' points: TL TR BR BL")
    return [list(p) for p in pts]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test the floor piano against a video file.")
    p.add_argument("video", help="path to an mp4 / video file")
    p.add_argument("--white", type=int, default=constants.DEFAULT_NUM_WHITE,
                   help="number of white keys (black keys are derived)")
    p.add_argument("--octave", type=int, default=constants.START_OCTAVE,
                   help="octave of the leftmost white key")
    p.add_argument("--floor", type=int, default=constants.DEFAULT_FLOOR_DEPTH,
                   help="depth (mm) the brightest pixels map to (the 'floor')")
    p.add_argument("--near", type=int, default=None,
                   help="depth (mm) the darkest pixels map to (default: floor-200)")
    p.add_argument("--threshold", type=int, default=constants.DEFAULT_TRIGGER_THRESHOLD,
                   help="trigger buffer (mm) above the floor")
    p.add_argument("--invert", action="store_true",
                   help="brightness mode: treat BRIGHT pixels as close instead of dark")
    p.add_argument("--motion", action="store_true",
                   help="detect the foot as what differs from the median background "
                        "(use this for ordinary RGB clips)")
    p.add_argument("--motion-threshold", type=int, default=35,
                   help="min per-channel diff vs background to count as foot")
    p.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                   help="pre-rotate the clip (rarely needed: auto-mat handles "
                        "orientation by itself)")
    p.add_argument("--corners", type=parse_corners, default=None,
                   help="manual mat corners 'x,y x,y x,y x,y' (TL TR BR BL)")
    p.add_argument("--full-frame", action="store_true",
                   help="use the whole video frame as the mat (old behaviour)")
    p.add_argument("--loop", action="store_true", help="replay the video on repeat")
    p.add_argument("--fps", type=float, default=0.0,
                   help="cap playback speed (0 = as fast as possible)")
    p.add_argument("--no-audio", action="store_true", help="run silently (no pygame)")
    p.add_argument("--show", action="store_true",
                   help="show the warped detection view (needs a display)")
    p.add_argument("--out", default=None,
                   help="write the detection visualisation to this MP4 file")
    return p.parse_args(argv)


class _SilentAudio:
    """No-op audio sink so --no-audio works without pygame/samples."""
    def update(self, active):
        pass

    def close(self):
        pass


def choose_corners(args, camera):
    """Corner source priority: --corners > --full-frame > auto-mat > full frame."""
    w, h = camera.width, camera.height
    full = [[0, 0], [w, 0], [w, h], [0, h]]
    if args.corners is not None:
        log.info("Using manual corners: %s", args.corners)
        return args.corners
    if args.full_frame:
        return full
    if camera.background is None:
        log.warning("No median background available — using the full frame as mat.")
        return full
    corners, info = mat_calibration.auto_calibrate(camera.background, args.white)
    if corners is None:
        log.warning("Mat auto-calibration failed (%s) — using the full frame. "
                    "Pass --corners to set the mat manually.", info.get("reason"))
        return full
    log.info("Mat auto-calibrated: black-key IoU %.2f, %s bars matched, "
             "grid residual %s -> %s px",
             info.get("black_key_iou", 0.0), info.get("matched_bars"),
             _fmt(info.get("residual_before")), _fmt(info.get("residual_after")))
    return [[float(x), float(y)] for x, y in corners]


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else "n/a"


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        camera = VideoDepthCamera(args.video, floor_depth=args.floor, near_depth=args.near,
                                  invert=args.invert, loop=args.loop, rotate=args.rotate,
                                  motion=args.motion, motion_threshold=args.motion_threshold,
                                  trigger_threshold=args.threshold)
        camera.start()
    except (DepthCameraError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)

    config = {
        "corners": choose_corners(args, camera),
        "num_white_keys": args.white,
        "start_octave": args.octave,
        "floor_depth": args.floor,
        "trigger_threshold": args.threshold,
        # keep the synthetic "foot" depth inside the press-height band even
        # when --near maps it far above the floor
        "max_press_height": max(constants.MAX_PRESS_HEIGHT,
                                args.floor - camera.near_depth + args.threshold),
    }
    audio = _SilentAudio() if args.no_audio else None  # None -> FloorPiano builds PianoAudio
    piano = FloorPiano(config, camera=camera, audio=audio)

    headless = args.show and not os.environ.get("DISPLAY")
    if headless:
        log.warning("--show needs a display ($DISPLAY unset); running without the window.")
    show = args.show and not headless

    writer = None
    if args.out:
        out_fps = args.fps if args.fps > 0 else (camera.fps if camera.fps else 25.0)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (piano.target_width, piano.target_height))
        if not writer.isOpened():
            log.error("could not open output video for writing: %s", args.out)
            sys.exit(1)
        log.info("Writing detection video to %s (%.1f fps)", args.out, out_fps)

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    log.info("Playing %s (floor=%dmm near=%dmm mode=%s) ...",
             args.video, args.floor, camera.near_depth,
             "motion" if args.motion else "brightness")
    n = 0
    events = []   # (frame_no, note) note-on events, printed as a summary at the end
    prev = set()
    try:
        while True:
            t0 = time.time()
            depth = camera.read_depth()
            if depth is None:
                break  # end of video (loop=False)
            n += 1
            warped, active = piano.process_frame(depth)
            for note in sorted(active - prev):
                events.append((n, note))
                log.info("frame %d NOTE ON: %s (active: %s)", n, note, sorted(active))
            prev = active
            if show or writer is not None:
                vis = render_frame(piano, warped, active, bgr=camera.last_bgr, frame_no=n)
                if writer is not None:
                    writer.write(vis)
                if show:
                    cv2.imshow("Floor Piano - video test", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
            if period:
                time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if writer is not None:
            writer.release()
        piano.shutdown()
    log.info("Done after %d frames. %d note-on events: %s", n, len(events),
             "  ".join(f"{note}@{f}" for f, note in events) or "(none)")


if __name__ == "__main__":
    main()
