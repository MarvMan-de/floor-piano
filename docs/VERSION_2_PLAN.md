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
*   **CPU-only depth pipeline**: The depth-threshold trigger runs comfortably on the Pi 5 CPU — no NPU required for the core piano.
*   **Realistic target**: ~30 FPS (Astra Pro depth-stream limit) and ~40–80 ms foot-to-audio latency.
*   **Optional / deferred — Hailo-8L pose**: A Hailo-8L could later add RGB pose estimation for richer multi-person tracking. This is **not** part of the current hardware plan; note that pose models expect 3-channel RGB, not raw single-channel depth, so an "Astra→Hailo depth fusion" would not offload 100% of the work.

## 📝 Implementation Roadmap (Next Steps)
1.  **[x] Phase 1: ArUco Integration**: `calibrate.py` auto-detects corners via `cv2.aruco`. Config saved to script dir (not CWD).
2.  **[~] Phase 1b: SDK Migration**: Code uses `pyorbbecsdk` via a single `depth_camera.py` (lazy import, Y16 request + size guard). **Not yet verified on the real Astra Pro** — the Pro may need the OpenNI2 backend (see CODE_REVIEW blocker #1). Must be tested on hardware.
3.  **[ ] Phase 2: 3D Plane Fitting**: Implement RANSAC plane fitting in place of current median floor sampling.
4.  **[~] Phase 3: Headless Service**: Done — headless calibrate auto-save, headless run loop, `SIGTERM` handling, logging, and a `systemd` unit template (`docs/floor-piano.service`). **Pending** — GPIO status LED + physical reset button.
5.  **[ ] Phase 4: Enclosure Design**: Design a 3D-printable box to house the Pi 5 and Astra Pro as a single unit.
6.  **[ ] Optional / deferred — Hailo-8L Pose**: Only if added later — wire an RGB pose model into a Hailo NPU for multi-person support. Not required for the core depth-based piano.

> ⚠️ **Real blocker before anything else:** the RGB→Depth registration (CODE_REVIEW #2). ArUco corners are found in the RGB image but the warp is applied to the depth image; the two sensors differ in FOV/origin, so the key mapping is off. This needs the real camera (D2C alignment or detecting markers on the depth-registered stream) and gates Phase 2 onward.

---
*Created on 2026-04-10 | Status: Drafting / Research Phase*
