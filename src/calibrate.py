"""Calibrate the floor piano: find the mat corners + measure the floor depth.

Two corner sources, tried in this order (selectable with --source):

  * aruco — printed ArUco markers IDs 0-3 at the mat corners (0=TL, 1=TR,
            2=BR, 3=BL as seen by the camera);
  * mat   — no markers: the printed piano mat itself is detected, its
            orientation resolved from the black-key pattern and the grid
            refined onto the painted keys (see mat_calibration.py).

The floor depth is sampled every frame at the mat centre and the saved value
is the median over the stable window — with a spread check, so a person
stepping through the scene (or standing on the mat) can't poison the
calibration. Keep the mat clear while calibrating.
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

# Resolve sibling imports regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants
import mat_calibration
import placement
from depth_camera import DepthCamera, DepthCameraError
from detection import build_config, sample_floor_depth, validate_config

log = logging.getLogger("calibrate")

# Consecutive good detections required before auto-saving in headless mode.
HEADLESS_STABLE_FRAMES = 15
# Floor samples within the stable window may spread at most this much (mm).
FLOOR_SPREAD_LIMIT = 60
# Give up (exit 1) if headless calibration hasn't locked within this time.
HEADLESS_TIMEOUT_S = 120.0


def find_rgb_camera(max_index=5):
    """Return an opened cv2.VideoCapture for the Astra's UVC RGB stream, or None.

    The Pi exposes several /dev/video* nodes (RGB, IR, metadata). We validate
    each candidate with an actual colour frame: V4L2 converts even mono/IR
    streams to 3-channel BGR, so a frame whose channels are all identical is a
    grayscale/IR node in disguise and is rejected. Set FLOOR_PIANO_RGB_INDEX to
    force a specific index if auto-detection misfires.
    """
    override = os.environ.get("FLOOR_PIANO_RGB_INDEX")
    indices = [int(override)] if override else range(max_index)
    for i in indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                if override or not ((b == g).all() and (g == r).all()):
                    log.info("Using RGB camera at index %d (%dx%d).",
                             i, frame.shape[1], frame.shape[0])
                    return cap
                log.info("Index %d delivers identical channels (IR/mono node) — skipping.", i)
        cap.release()  # not a usable colour camera; don't leak the handle
    return None


def detect_corners_aruco(detector, rgb_frame):
    """Return the 4 marker centres as a TL,TR,BR,BL float array, else None.

    The marker *centre* is used as the corner reference. This insets the playable
    area by ~half a marker uniformly on all sides (fine for play); see SETUP.md.
    """
    corners, ids, _ = detector.detectMarkers(rgb_frame)
    if ids is None or len(ids) < 4:
        return None
    found = {}
    for i in range(len(ids)):
        mid = int(ids[i][0])
        if mid in constants.CORNER_IDS:
            found[mid] = np.mean(corners[i][0], axis=0)
    if not all(mid in found for mid in constants.CORNER_IDS):
        return None
    return np.float32([found[mid] for mid in constants.CORNER_IDS])


def save_config(config_data, path):
    with open(path, "w") as f:
        json.dump(config_data, f, indent=4)
    log.info("Calibration saved to %s (floor_depth=%dmm).", path, config_data["floor_depth"])


def write_placement_status(path, status, stable, needed):
    """Persist the live placement verdict for the Pi-hotspot web UI to poll.

    Best-effort: a write failure must never interrupt calibration. The web UI
    (built separately) reads this JSON; it can also import placement.assess_placement
    directly. See src/placement.py for the field contract.
    """
    data = placement.status_to_dict(status)
    data["stable_frames"] = stable
    data["needed_frames"] = needed
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _annotate(rgb, corners, floor_depth, method):
    pts = corners.astype(int)
    cv2.polylines(rgb, [pts], True, (0, 255, 0), 2)
    for i, pt in enumerate(pts):
        cv2.circle(rgb, tuple(pt), 8, (0, 255, 0), -1)
        cv2.putText(rgb, "TL TR BR BL".split()[i], tuple(pt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    floor_txt = f"{floor_depth}mm" if floor_depth is not None else "no depth!"
    cv2.putText(rgb, f"[{method}] Floor: {floor_txt}  -  press 's' to save",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calibrate the floor piano.")
    p.add_argument("--source", choices=("auto", "aruco", "mat"), default="auto",
                   help="corner source: ArUco markers, the printed mat itself, "
                        "or auto (ArUco first, mat as fallback)")
    p.add_argument("--white", type=int, default=constants.DEFAULT_NUM_WHITE,
                   help="number of white keys on the mat")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_output = os.path.join(script_dir, "config.json")
    status_output = os.path.join(script_dir, "placement_status.json")
    headless = not os.environ.get('DISPLAY')  # unset OR empty -> headless

    camera = DepthCamera()
    try:
        camera.start()
    except DepthCameraError as e:
        log.error("%s", e)
        sys.exit(1)

    cap = find_rgb_camera()
    if cap is None:
        log.error("Astra RGB camera not found.")
        camera.stop()
        sys.exit(1)

    detector = None
    if args.source in ("auto", "aruco"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    log.info("--- PIANO CALIBRATION (%s mode, source=%s) ---",
             "headless" if headless else "GUI", args.source)
    log.info("Keep the mat clear of feet. ArUco IDs 0-3 (0=TL 1=TR 2=BR 3=BL) "
             "are used when present; otherwise the mat itself is detected.")
    if headless:
        log.info("Headless: auto-saving once detection stays stable for %d frames "
                 "(timeout %.0fs).", HEADLESS_STABLE_FRAMES, HEADLESS_TIMEOUT_S)
    else:
        log.info("Press 's' to save, 'q' to quit.")

    stable = 0
    floor_samples = []
    started = time.monotonic()
    last_progress = started
    last_status_write = 0.0
    rgb_fail = 0
    try:
        while True:
            if headless and time.monotonic() - started > HEADLESS_TIMEOUT_S:
                log.error("Calibration did not lock within %.0fs — giving up.",
                          HEADLESS_TIMEOUT_S)
                sys.exit(1)
            ret, rgb = cap.read()
            if not ret:
                rgb_fail += 1
                if rgb_fail >= 100:
                    log.error("RGB camera stopped delivering frames — giving up.")
                    sys.exit(1)
                time.sleep(0.05)
                continue
            rgb_fail = 0
            depth = camera.read_depth()
            if depth is None:
                continue

            corners, method = None, None
            if detector is not None:
                corners = detect_corners_aruco(detector, rgb)
                method = "aruco" if corners is not None else None
            if corners is None and args.source in ("auto", "mat"):
                corners, info = mat_calibration.auto_calibrate(rgb, args.white)
                if corners is not None:
                    method = "mat"

            config_data = None
            floor_depth = None
            if corners is not None:
                center = (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))
                floor_depth = sample_floor_depth(depth, center, rgb.shape)
                if floor_depth is None:
                    log.warning("Markers/mat found but no valid depth at the mat centre "
                                "— is the mat inside the depth camera's view?")
                    stable, floor_samples = 0, []
                else:
                    floor_samples.append(floor_depth)
                    window = floor_samples[-HEADLESS_STABLE_FRAMES:]
                    if max(window) - min(window) > FLOOR_SPREAD_LIMIT:
                        # Someone/something moving through the sample area.
                        stable, floor_samples = 0, [floor_depth]
                    else:
                        stable += 1
                    corner_points = {mid: corners[i]
                                     for i, mid in enumerate(constants.CORNER_IDS)}
                    config_data = build_config(corner_points, int(np.median(window)),
                                               rgb.shape, num_white=args.white)
                    # Never count (or save) a config main.py would reject — e.g.
                    # a mirrored/degenerate corner quad from a bad mat detection.
                    try:
                        validate_config(config_data)
                    except ValueError as e:
                        log.warning("Rejecting unusable detection (%s): %s", method, e)
                        config_data = None
                        stable, floor_samples = 0, []
                    if not headless and config_data is not None:
                        _annotate(rgb, corners, floor_depth, method)
            else:
                stable, floor_samples = 0, []

            # Live placement verdict (pure geometry) — drives the headless coach
            # log, the GUI overlay, and the web-UI status file from one source.
            status = placement.assess_placement(
                corners, rgb.shape, depth_ok=floor_depth is not None)
            now = time.monotonic()
            if now - last_status_write >= 0.3:
                write_placement_status(status_output, status, stable, HEADLESS_STABLE_FRAMES)
                last_status_write = now
            if not headless:
                cv2.putText(rgb, status.headline, (20, rgb.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(rgb, status.hint, (20, rgb.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if headless:
                if now - last_progress >= 5.0:
                    log.info("[placement] %s — %s (stable %d/%d)", status.headline,
                             status.hint, stable, HEADLESS_STABLE_FRAMES)
                    last_progress = now
                if config_data is not None and stable >= HEADLESS_STABLE_FRAMES:
                    save_config(config_data, config_output)
                    break
                continue

            cv2.imshow("Calibration", rgb)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and config_data is not None:
                save_config(config_data, config_output)
                break
            elif key == ord('q'):
                log.info("Quit without saving.")
                break
    finally:
        cap.release()
        camera.stop()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
