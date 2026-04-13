# 🎹 Floor Piano v2.0 (Professional Implementation)

This system is specifically optimized for the **Raspberry Pi 5 + Orbbec Astra Pro + Hailo-8L** hardware stack. It leverages hardware-accelerated 3D depth sensing and NPU-based pose estimation for ultra-low latency (<10ms) and high-precision interaction.

## 🛠 Required Hardware

- **Raspberry Pi 5** (8GB RAM Recommended)
- **Raspberry Pi AI Kit** (Hailo-8L M.2 AI Module + M.2 HAT+)
- **Official Raspberry Pi 27W Power Supply** (Critical for Astra + Hailo power draw)
- **Orbbec Astra Pro 3D Camera** (USB 3.0)
- **USB Audio Adapter or USB Speakers** (Pi 5 lacks a 3.5mm jack)

## 🚀 Installation

### 1. System-Level Dependencies
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-pip libopencv-dev libatlas-base-dev libportaudio2 libopenni2-dev ffmpeg
```

### 2. Hailo-8L NPU Setup
```bash
sudo apt install hailo-all -y
sudo reboot
```

### 3. Python Environment Setup
```bash
pip3 install -r requirements.txt
```

### 4. Audio Samples Initialization
```bash
cd workspace/projects/floor-piano/sounds
chmod +x download_samples.sh
./download_samples.sh
```

## 📐 Automatic Calibration (Zero-UI)

Version 2.0 uses **ArUco Markers** for autonomous calibration. This allows the system to be "Headless" (no monitor required).

1.  **Prepare Markers**: Print 4 ArUco markers from the `DICT_4X4_50` dictionary (IDs 0, 1, 2, 3).
2.  **Place Markers**: Place them at the 4 corners of your "piano" area on the floor.
    *   **ID 0**: Top-Left
    *   **ID 1**: Top-Right
    *   **ID 2**: Bottom-Right
    *   **ID 3**: Bottom-Left
3.  **Run Calibration**:
    ```bash
    python3 calibrate.py
    ```
4.  The system will automatically detect the markers, calculate the perspective warp, and sample the floor's depth to establish the "Trigger Zone." Press 's' to save.

## 🎹 Running the Piano

```bash
python3 main.py
```

### Key Features:
*   **3D Depth Triggering**: Notes are triggered when a foot enters the 50mm (5cm) "active zone" above the floor.
*   **Auto-Leveling**: Press 'r' to re-sample the floor plane if the camera is bumped or the floor surface changes.
*   **Headless Design**: Once calibrated, the system can be set to run as a `systemd` service on boot.

## 🔧 Performance & Optimization
*   **90 FPS Target**: The system is designed to maintain 90 FPS inference and depth processing.
*   **Latency**: The 3D trigger bypasses complex AI pipelines for the initial "hit," resulting in <2ms detection latency and ~6ms audio latency.
*   **Power**: If you see "Under-voltage" warnings, the Astra Pro will likely disconnect. Ensure you are using the **Official 27W PSU**.
