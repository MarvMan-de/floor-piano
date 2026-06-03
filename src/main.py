import cv2
import json
import numpy as np
import time
import os
import sys

# Resolve imports regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio import PianoAudio

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType
except ImportError:
    print("Fatal Error: 'pyorbbecsdk' not found. Install with 'pip install pyorbbecsdk'.")
    sys.exit(1)

try:
    import hailo
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False

class FloorPianoV2:
    def __init__(self, config_path=None):
        # Resolve config relative to this script, not CWD
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config.json")

        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found. Run 'python calibrate.py' first.")
            sys.exit(1)
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.audio = PianoAudio()

        self.corners = np.float32(self.config["corners"])
        self.target_width = 700
        self.target_height = 200
        self.dst_pts = np.float32([[0, 0], [self.target_width, 0],
                                   [self.target_width, self.target_height], [0, self.target_height]])
        self.M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

        self.keys = self.config["keys"]
        self.floor_depth = self.config.get("floor_depth", 1000)
        self.threshold = self.config.get("trigger_threshold", 50)  # 50mm (5cm) buffer

        # Initialize Orbbec pipeline
        try:
            self.pipeline = Pipeline()
            orbbec_config = Config()
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            orbbec_config.enable_stream(depth_profile)
            self.pipeline.start(orbbec_config)
            print("Orbbec Astra Pro Depth Stream: ACTIVE")
        except Exception as e:
            print(f"Astra Init Error: {e}")
            sys.exit(1)

        if HAS_HAILO:
            print("Hailo-8L NPU: DETECTED")

    def auto_level_floor(self, depth_array):
        """Re-samples floor depth from current scene."""
        warped_depth = cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height))
        valid_depths = warped_depth[warped_depth > 0]
        if len(valid_depths) > 1000:
            new_floor = np.median(valid_depths)
            self.floor_depth = int(0.9 * self.floor_depth + 0.1 * new_floor)

    def run(self):
        print("\n--- FLOOR PIANO v2.0 ---")
        print(f"Floor Depth: {self.floor_depth}mm | Trigger: <{self.floor_depth - self.threshold}mm")
        print("Press 'q' to quit, 'r' to re-calibrate floor level.")

        key_width = self.target_width // len(self.keys)

        try:
            while True:
                start_time = time.time()

                frames = self.pipeline.wait_for_frames(100)
                depth_frame = frames.get_depth_frame() if frames else None
                if depth_frame is None:
                    continue

                h = depth_frame.get_height()
                w = depth_frame.get_width()
                depth_array = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))

                warped_depth = cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height))
                hits = (warped_depth > 0) & (warped_depth < (self.floor_depth - self.threshold))

                current_active = set()
                if np.any(hits):
                    for i in range(len(self.keys)):
                        key_zone = hits[:, i*key_width : (i+1)*key_width]
                        if np.sum(key_zone) > 150:  # pixel threshold to filter noise
                            current_active.add(self.keys[i])

                self.audio.update(current_active)

                if 'DISPLAY' in os.environ:
                    vis_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                    for i in range(len(self.keys)):
                        color = (0, 255, 0) if self.keys[i] in current_active else (100, 100, 100)
                        cv2.rectangle(vis_frame, (i*key_width, 0), ((i+1)*key_width, self.target_height), color, 2)
                        cv2.putText(vis_frame, self.keys[i], (i*key_width + 10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    vis_frame[hits] = [0, 0, 255]
                    cv2.imshow("Floor Piano v2.0 - 3D View", vis_frame)

                fps = 1.0 / (time.time() - start_time)
                if int(fps) % 30 == 0:
                    sys.stdout.write(f"\rFPS: {fps:.1f} | Floor: {self.floor_depth}mm | Active: {list(current_active)}")
                    sys.stdout.flush()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\nRe-leveling floor...")
                    self.auto_level_floor(depth_array)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.pipeline.stop()
            self.audio.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FloorPianoV2()
    app.run()
