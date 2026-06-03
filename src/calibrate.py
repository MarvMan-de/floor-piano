import cv2
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType
except ImportError:
    print("Fatal Error: 'pyorbbecsdk' not found. Install with 'pip install pyorbbecsdk'.")
    sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_output = os.path.join(script_dir, "config.json")

    # Orbbec: depth via pyorbbecsdk
    try:
        pipeline = Pipeline()
        orbbec_config = Config()
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        orbbec_config.enable_stream(depth_profile)
        pipeline.start(orbbec_config)
        print("Orbbec Astra Depth Stream Started.")
    except Exception as e:
        print(f"Error initializing Orbbec: {e}")
        sys.exit(1)

    # Astra Pro RGB is a standard UVC camera — find it via OpenCV
    cap = None
    for i in range(5):
        candidate = cv2.VideoCapture(i)
        if candidate.isOpened():
            cap = candidate
            print(f"Astra RGB found on index {i}")
            break
    if cap is None:
        print("Error: Astra RGB camera not found.")
        pipeline.stop()
        sys.exit(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("\n--- PIANO CALIBRATION ---")
    print("1. Place ArUco markers (ID 0-3) at the 4 corners of the mat.")
    print("   Order: 0=Top-Left, 1=Top-Right, 2=Bottom-Right, 3=Bottom-Left")
    print("2. System auto-detects corners and samples floor depth.")
    print("Press 's' to save, 'q' to quit.")

    corners_found = None
    floor_depth = 1000

    try:
        while True:
            ret, rgb_frame = cap.read()
            if not ret:
                continue

            frames = pipeline.wait_for_frames(100)
            depth_frame = frames.get_depth_frame() if frames else None
            if depth_frame is None:
                continue

            dh = depth_frame.get_height()
            dw = depth_frame.get_width()
            depth_array = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))

            corners, ids, _ = detector.detectMarkers(rgb_frame)

            if ids is not None and len(ids) >= 4:
                corner_pts = {}
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    if marker_id in [0, 1, 2, 3]:
                        corner_pts[marker_id] = np.mean(corners[i][0], axis=0)

                if all(mid in corner_pts for mid in [0, 1, 2, 3]):
                    pts = np.float32([corner_pts[0], corner_pts[1], corner_pts[2], corner_pts[3]])
                    corners_found = pts.tolist()

                    for i, pt in enumerate(pts):
                        cv2.circle(rgb_frame, tuple(pt.astype(int)), 10, (0, 255, 0), -1)
                        cv2.putText(rgb_frame, f"ID {i}", tuple(pt.astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Scale RGB coordinates to depth resolution for sampling
                    rh, rw = rgb_frame.shape[:2]
                    center_x = int(np.mean(pts[:, 0]))
                    center_y = int(np.mean(pts[:, 1]))
                    dx = int(center_x * dw / rw)
                    dy = int(center_y * dh / rh)

                    if 10 <= dy < dh - 10 and 10 <= dx < dw - 10:
                        sample = depth_array[dy-10:dy+10, dx-10:dx+10]
                        valid = sample[sample > 0]
                        if len(valid) > 0:
                            floor_depth = int(np.median(valid))
                            cv2.putText(rgb_frame, f"Floor Depth: {floor_depth}mm", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.putText(rgb_frame, "Press 's' to save", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(rgb_frame, "Searching for ArUco markers (IDs 0-3)...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Calibration", rgb_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and corners_found:
                rh, rw = rgb_frame.shape[:2]
                config_data = {
                    "corners": corners_found,
                    "keys": ["C", "D", "E", "F", "G", "A", "B"],
                    "floor_depth": floor_depth,
                    "trigger_threshold": 50,
                    "canvas_size": [rw, rh]
                }
                with open(config_output, "w") as f:
                    json.dump(config_data, f, indent=4)
                print(f"Calibration saved to {config_output}. Floor depth: {floor_depth}mm.")
                break
            elif key == ord('q'):
                break

    finally:
        cap.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
