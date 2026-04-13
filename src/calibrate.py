import cv2
import json
import numpy as np
import os
import sys
import time

# Try to import OpenNI2 for Orbbec Astra
try:
    from openni import openni2
    HAS_OPENNI = True
except ImportError:
    HAS_OPENNI = False
    print("Error: openni library not found. Install it with 'pip install openni'.")

def main():
    if not HAS_OPENNI:
        print("Hardware required: Orbbec Astra (OpenNI2).")
        sys.exit(1)

    # Initialize Astra
    try:
        openni2.initialize()
        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        print("Orbbec Astra Depth Stream Started.")
    except Exception as e:
        print(f"Error initializing Astra: {e}")
        sys.exit(1)

    # Initialize RGB Stream (Standard UVC)
    cap = cv2.VideoCapture(0) # Astra Pro RGB is usually the first camera
    if not cap.isOpened():
        print("Error: Could not open Astra RGB camera.")
        # We can still proceed with depth only if needed, but ArUco needs RGB
        # Let's try to find the right index if 0 fails
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Astra RGB found on index {i}")
                break
        else:
            print("Astra RGB camera not found.")
            sys.exit(1)

    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("\n--- 🎹 AUTOMATIC PIANO CALIBRATION ---")
    print("1. Place ArUco markers (ID 0-3) at the 4 corners of the mat.")
    print("2. The system will automatically detect the corners and sample depth.")
    print("Press 's' to save and exit, 'q' to quit.")

    corners_found = None
    floor_depth = 1000

    try:
        while True:
            # 1. Read RGB Frame
            ret, rgb_frame = cap.read()
            if not ret:
                break

            # 2. Read Depth Frame
            d_frame = depth_stream.read_frame()
            d_frame_data = d_frame.get_buffer_as_uint16()
            depth_array = np.ndarray((d_frame.height, d_frame.width), dtype=np.uint16, buffer=d_frame_data)

            # 3. Detect ArUco
            corners, ids, rejected = detector.detectMarkers(rgb_frame)
            
            if ids is not None and len(ids) >= 4:
                # We need IDs 0, 1, 2, 3
                corner_pts = {}
                for i in range(len(ids)):
                    corner_pts[ids[i][0]] = np.mean(corners[i][0], axis=0)
                
                if all(id in corner_pts for id in [0, 1, 2, 3]):
                    # Found all 4 corners
                    # Order: 0:TL, 1:TR, 2:BR, 3:BL
                    pts = np.float32([corner_pts[0], corner_pts[1], corner_pts[2], corner_pts[3]])
                    corners_found = pts.tolist()
                    
                    # Draw for debug
                    for i, pt in enumerate(pts):
                        cv2.circle(rgb_frame, tuple(pt.astype(int)), 10, (0, 255, 0), -1)
                        cv2.putText(rgb_frame, f"ID {i}", tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 4. Sample Depth in the middle of the mat
                    # Simple median of the center area of the detected mat
                    # Warp for depth is tricky if RGB and Depth aren't aligned perfectly
                    # For Astra Pro, they are separate. We use the RGB corners.
                    
                    # Just sample a few points around the center for now
                    center_x = int(np.mean(pts[:, 0]))
                    center_y = int(np.mean(pts[:, 1]))
                    
                    if 0 <= center_x < depth_array.shape[1] and 0 <= center_y < depth_array.shape[0]:
                        sample_depth = depth_array[center_y-10:center_y+10, center_x-10:center_x+10]
                        median_depth = np.median(sample_depth[sample_depth > 0])
                        if not np.isnan(median_depth):
                            floor_depth = int(median_depth)
                            cv2.putText(rgb_frame, f"Floor Depth: {floor_depth}mm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Calibration (RGB)", rgb_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and corners_found:
                # Save to config.json
                config = {
                    "corners": corners_found,
                    "keys": ["C", "D", "E", "F", "G", "A", "B"],
                    "floor_depth": floor_depth,
                    "trigger_threshold": 50, # 5cm above floor
                    "canvas_size": [rgb_frame.shape[1], rgb_frame.shape[0]]
                }
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Calibration saved to config.json! Floor depth set to {floor_depth}mm.")
                break
            elif key == ord('q'):
                break

    finally:
        cap.release()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
