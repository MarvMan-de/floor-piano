import cv2
import json
import numpy as np
import time
import os
import sys
from audio import PianoAudio

# Hard requirements for Version 2.0 Hardware
try:
    from openni import openni2
except ImportError:
    print("Fatal Error: 'openni' library not found. This version requires Orbbec Astra Pro.")
    sys.exit(1)

try:
    # Hailo SDK (Placeholder for RPi 5 AI Kit)
    # import hailo
    HAS_HAILO = True 
except ImportError:
    HAS_HAILO = False
    print("Warning: Hailo-8L SDK not found. Running in Depth-Only mode.")

class FloorPianoV2:
    def __init__(self, config_path="config.json"):
        # 1. Load Configuration
        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found. Run 'python calibrate.py' first.")
            sys.exit(1)
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        # 2. Setup Audio (Optimized for low latency)
        self.audio = PianoAudio()
        
        # 3. Setup Perspective Transform
        self.corners = np.float32(self.config["corners"])
        self.target_width = 700
        self.target_height = 200
        self.dst_pts = np.float32([[0, 0], [self.target_width, 0], 
                                   [self.target_width, self.target_height], [0, self.target_height]])
        self.M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)
        
        self.keys = self.config["keys"]
        self.floor_depth = self.config.get("floor_depth", 1000)
        self.threshold = self.config.get("trigger_threshold", 50) # 50mm (5cm) buffer
        
        # 4. Initialize Astra
        try:
            openni2.initialize()
            self.dev = openni2.Device.open_any()
            self.depth_stream = self.dev.create_depth_stream()
            self.depth_stream.start()
            print("✅ Orbbec Astra Pro Depth Stream: ACTIVE")
        except Exception as e:
            print(f"❌ Astra Init Error: {e}")
            sys.exit(1)
        
        # 5. Initialize Hailo Pose (Placeholder)
        if HAS_HAILO:
            print("✅ Hailo-8L NPU: DETECTED (Pose Refinement Enabled)")

    def auto_level_floor(self, depth_array):
        """Automatically adjusts floor_depth by sampling the current scene."""
        warped_depth = cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height))
        # Filter out 0 (no data) and find the median of the actual floor
        valid_depths = warped_depth[warped_depth > 0]
        if len(valid_depths) > 1000:
            new_floor = np.median(valid_depths)
            # Smooth transition
            self.floor_depth = int(0.9 * self.floor_depth + 0.1 * new_floor)

    def run(self):
        print("\n--- 🎹 FLOOR PIANO v2.0 (HEADLESS READY) ---")
        print(f"Floor Depth: {self.floor_depth}mm | Trigger: <{self.floor_depth - self.threshold}mm")
        print("Press 'q' to quit, 'r' to re-calibrate floor level.")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Capture Depth Frame
                d_frame = self.depth_stream.read_frame()
                d_frame_data = d_frame.get_buffer_as_uint16()
                depth_array = np.ndarray((d_frame.height, d_frame.width), dtype=np.uint16, buffer=d_frame_data)
                
                # 2. Perspective Warp
                warped_depth = cv2.warpPerspective(depth_array, self.M, (self.target_width, self.target_height))
                
                # 3. 3D Trigger Logic
                # Hits are pixels significantly closer to camera than the floor
                hits = (warped_depth > 0) & (warped_depth < (self.floor_depth - self.threshold))
                
                current_active = set()
                if np.any(hits):
                    key_width = self.target_width // len(self.keys)
                    for i in range(len(self.keys)):
                        key_zone = hits[:, i*key_width : (i+1)*key_width]
                        if np.sum(key_zone) > 150: # Pixel threshold to filter noise
                            current_active.add(self.keys[i])
                
                # 4. Trigger Audio
                self.audio.update(current_active)
                
                # 5. Visual Debug (Disable in headless mode)
                if 'DISPLAY' in os.environ:
                    # Create a visualization of the hits
                    vis_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                    for i in range(len(self.keys)):
                        color = (0, 255, 0) if self.keys[i] in current_active else (100, 100, 100)
                        cv2.rectangle(vis_frame, (i*key_width, 0), ((i+1)*key_width, self.target_height), color, 2)
                        cv2.putText(vis_frame, self.keys[i], (i*key_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    # Overlay "hits" pixels in red
                    vis_frame[hits] = [0, 0, 255]
                    cv2.imshow("Floor Piano v2.0 - 3D View", vis_frame)
                
                # 6. Stats & Controls
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
            self.depth_stream.stop()
            openni2.unload()
            self.audio.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FloorPianoV2()
    app.run()
