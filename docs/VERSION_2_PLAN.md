# 🎹 Floor Piano v2.0: The "Black Box" (Headless) Edition

## 🎯 Vision
Transform the current monitor-dependent prototype into a professional, portable, and autonomous "Zero-UI" interactive installation. The user should simply plug in the box, lay down the mat, and start playing within 5 seconds without ever seeing a screen.

## 🛠 Core Architectural Pillars

### 1. 🤖 Automatic Mat Detection (Zero-UI Calibration)
Instead of manual corner clicking, the system will use Computer Vision to "find" the playable area.
*   **ArUco Markers**: Place four unique ArUco codes (e.g., ID 0-3) on the corners of the mat.
*   **Auto-Perspective Warp**: The Pi 5 scans for these markers on boot. Once found, it automatically calculates the perspective transformation (the "warp") to define the piano keys.
*   **Fallback - Color Segmentation**: If markers aren't used, detect the mat's high-contrast border (e.g., Neon Green on a dark floor).

### 2. 🌊 Autonomous 3D Floor Leveling
Leverage the Orbbec Astra Pro's depth sensor for "Self-Correcting" calibration.
*   **Plane Fitting**: Instead of a median depth, use a RANSAC-based plane fitting algorithm to identify the exact angle and height of the floor surface inside the mat area.
*   **Dynamic Z-Trigger**: Automatically set the `floor_depth` based on the plane fit. If the camera is bumped or tilted, the system will re-calculate the plane in real-time without stopping the music.
*   **5cm Safety Buffer**: Establish a 50mm "trigger zone" above the detected plane to ensure only feet (not the floor itself) trigger notes.

### 3. 📦 Headless Infrastructure
Eliminate the need for a keyboard, mouse, and monitor.
*   **Systemd Service**: Create a background service (`floor-piano.service`) that launches the piano engine at boot.
*   **Status Indicators (GPIO)**: Add a simple RGB LED to the "Piano Box":
    *   🔵 **Blinking Blue**: System booting / Initializing Astra.
    *   🟡 **Blinking Yellow**: Searching for Mat/ArUco Markers.
    *   🟢 **Solid Green**: Calibrated and Ready to Play.
    *   🔴 **Solid Red**: Error (Camera disconnected / Power insufficient).
*   **Physical Reset Button**: A hardware button on the box to clear the current `config.json` and force a fresh "Auto-Calibration" scan.

### 4. ⚡️ Performance Optimization
*   **Hailo-8L + Astra Fusion**: Direct the Astra's 3D depth stream into the Hailo-8L's pose estimation pipeline to offload 100% of the Vision/AI processing from the Pi 5's CPU.
*   **90 FPS Target**: Maintain a rock-solid 90 FPS for near-zero latency (under 10ms), matching the response time of a high-end digital piano.

## 📝 Implementation Roadmap (Next Steps)
1.  **[x] Phase 1: ArUco Integration**: `calibrate.py` auto-detects corners via `cv2.aruco`. Config saved to script dir (not CWD).
2.  **[x] Phase 1b: SDK Migration**: Replaced dead `openni`/OpenNI2 bindings with official `pyorbbecsdk` (OrbbecSDK). Fixed all CWD-relative path bugs.
3.  **[ ] Phase 2: 3D Plane Fitting**: Implement RANSAC plane fitting in place of current median floor sampling.
4.  **[ ] Phase 3: Hailo-8L Pose Integration**: Wire foot detection model into Hailo NPU for multi-person support and sub-pixel accuracy.
5.  **[ ] Phase 4: Headless Service**: Create `systemd` service and Python GPIO scripts for LED/Button control.
6.  **[ ] Phase 5: Enclosure Design**: Design a 3D-printable box to house the Pi 5, Hailo Kit, and Astra Pro as a single unit.

---
*Created on 2026-04-10 | Status: Drafting / Research Phase*
