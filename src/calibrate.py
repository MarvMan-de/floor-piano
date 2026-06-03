import json
import logging
import os
import sys

import cv2
import numpy as np

# Resolve sibling imports regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants
from depth_camera import DepthCamera, DepthCameraError
from detection import build_config, sample_floor_depth

log = logging.getLogger("calibrate")

# Consecutive good detections required before auto-saving in headless mode.
HEADLESS_STABLE_FRAMES = 15


def find_rgb_camera(max_index=5):
    """Return an opened cv2.VideoCapture for the Astra's UVC RGB stream, or None.

    The Pi exposes several /dev/video* nodes (RGB, IR, metadata). We validate each
    candidate with an actual 3-channel colour frame instead of trusting isOpened().
    Set FLOOR_PIANO_RGB_INDEX to force a specific index if auto-detection misfires.
    """
    override = os.environ.get("FLOOR_PIANO_RGB_INDEX")
    indices = [int(override)] if override else range(max_index)
    for i in indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                log.info("Using RGB camera at index %d (%dx%d).", i, frame.shape[1], frame.shape[0])
                return cap
        cap.release()  # not a usable colour camera; don't leak the handle
    return None


def detect_corners(detector, rgb_frame):
    """Return {id: (x, y) marker centre} for all 4 corner markers, else None.

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
    if all(mid in found for mid in constants.CORNER_IDS):
        return found
    return None


def save_config(config_data, path):
    with open(path, "w") as f:
        json.dump(config_data, f, indent=4)
    log.info("Calibration saved to %s (floor_depth=%dmm).", path, config_data["floor_depth"])


def _annotate(rgb, found, floor_depth):
    for mid in constants.CORNER_IDS:
        pt = tuple(found[mid].astype(int))
        cv2.circle(rgb, pt, 10, (0, 255, 0), -1)
        cv2.putText(rgb, f"ID {mid}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(rgb, f"Floor: {floor_depth}mm  -  press 's' to save",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_output = os.path.join(script_dir, "config.json")
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

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    log.info("--- PIANO CALIBRATION (%s mode) ---", "headless" if headless else "GUI")
    log.info("Place ArUco markers IDs 0-3 at the mat corners (0=TL, 1=TR, 2=BR, 3=BL).")
    if headless:
        log.info("Headless: auto-saving once markers stay detected for %d frames.",
                 HEADLESS_STABLE_FRAMES)
    else:
        log.info("Press 's' to save, 'q' to quit.")

    stable = 0
    try:
        while True:
            ret, rgb = cap.read()
            if not ret:
                continue
            depth = camera.read_depth()
            if depth is None:
                continue

            found = detect_corners(detector, rgb)
            config_data = None
            if found is not None:
                center = (float(np.mean([found[m][0] for m in constants.CORNER_IDS])),
                          float(np.mean([found[m][1] for m in constants.CORNER_IDS])))
                floor_depth = sample_floor_depth(depth, center, rgb.shape)
                config_data = build_config(found, floor_depth, rgb.shape)
                stable += 1
                if not headless:
                    _annotate(rgb, found, floor_depth)
            else:
                stable = 0
                if not headless:
                    cv2.putText(rgb, "Searching for ArUco markers (IDs 0-3)...",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if headless:
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
