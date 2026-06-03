# 🎹 Floor Piano v2.0 (Professional Implementation)

This system targets the **Raspberry Pi 5 + Orbbec Astra Pro** hardware stack. It uses 3D depth sensing to detect feet above the floor plane, which keeps false triggers low without any AI/NPU processing.

## 🛠 Required Hardware

- **Raspberry Pi 5** (8GB RAM Recommended)
- **Official Raspberry Pi 27W Power Supply** (Recommended — the Pi 5's USB ports share a limited current budget, and an under-powered Astra can disconnect)
- **Orbbec Astra Pro 3D Camera**
- **USB Audio Adapter or USB Speakers** (Pi 5 lacks a 3.5mm jack)

## 🚀 Installation

### 1. System-Level Dependencies
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-pip libopencv-dev libatlas-base-dev libportaudio2 ffmpeg libusb-1.0-0-dev
```

### 2. Python Environment Setup
```bash
pip3 install -r requirements.txt
```

> ⚠️ **pyorbbecsdk on the Pi (arm64):** a plain `pip install pyorbbecsdk` usually does
> **not** work. You typically have to build the OrbbecSDK Python bindings from source
> and install the udev rules so the camera is accessible without root. Verify the
> camera is seen first with `lsusb`. If the depth stream cannot be opened, the app now
> exits with a clear message instead of a stack trace.
>
> **Note on the Astra _Pro_:** the original Astra Pro is a Legacy OpenNI2/UVC device.
> If `pyorbbecsdk` reports no depth profile on your unit, depth must be read via the
> OpenNI2 backend instead — see CODE_REVIEW.md (blocker #1). Test this first on the
> real hardware.

### 3. Audio Samples Initialization
Generate the note samples locally (reliable, offline — no external downloads):
```bash
cd src/sounds
python3 generate_samples.py
```
(Optional: `./download_samples.sh` fetches real piano samples instead, but it needs a valid source URL — the old default is dead.)

## 🧪 Testing Without the Camera

You can verify almost everything except the camera itself, in this order:

1. **Logic (numpy only):**
   ```bash
   pip install -r requirements-dev.txt && pytest
   ```
   Detection, geometry, decode and config logic — 51 tests.
2. **Samples (stdlib only):**
   ```bash
   python3 src/sounds/generate_samples.py
   ```
3. **Audio device (pygame):** confirm your USB speaker works.
   ```bash
   python3 src/sounds/play_test.py    # should play C D E F G A B
   ```
4. **Full pipeline minus camera (cv2 + pygame):** a synthetic foot sweeps the keys —
   you should *hear* the scale. This proves warp → detect → audio end-to-end.
   ```bash
   python3 src/demo_mock.py
   ```
5. **Print the markers (cv2):** prepare for on-site calibration.
   ```bash
   python3 tools/generate_aruco.py    # writes markers/marker_0..3.png
   ```

Only the Orbbec depth stream (`calibrate.py` / `main.py`) and the RGB→depth
registration need the real camera.

## 📷 Camera Placement

The Astra Pro is a structured-light depth camera — mind its physics:

*   **Minimum range ~0.6 m.** Anything closer reads as 0 (invalid). Mount high enough.
*   **Field of view.** Looking straight down from ~1.0 m covers only ~1.1 m width. For
    7 keys + feet, mount higher (e.g. 1.5–2.0 m) or accept a narrower mat.
*   **Lighting.** Structured-light IR is disturbed by direct sunlight; dark/glossy floors
    absorb IR and create holes (0-pixels) that read as "no foot". Prefer indoor, matte,
    light-coloured surfaces.
*   **Calibration marker convention.** The corner is taken as the **centre** of each
    ArUco marker, so place the markers' centres exactly on the 4 playable-area corners
    (the playable zone ends up inset by ~half a marker — this is expected).

## 📐 Automatic Calibration (ArUco)

Calibration uses **ArUco Markers**. It runs in a GUI window (press `s` to save) or, if no
display is attached, **headless** — it auto-saves once all 4 markers stay detected for a
short, stable period.

1.  **Prepare Markers**: Print 4 ArUco markers from the `DICT_4X4_50` dictionary (IDs 0, 1, 2, 3).
2.  **Place Markers**: Place them at the 4 corners of your "piano" area on the floor.
    *   **ID 0**: Top-Left
    *   **ID 1**: Top-Right
    *   **ID 2**: Bottom-Right
    *   **ID 3**: Bottom-Left
3.  **Run Calibration** (from the repo root):
    ```bash
    python3 src/calibrate.py
    ```
4.  The system will automatically detect the markers, calculate the perspective warp, and sample the floor's depth to establish the "Trigger Zone." Press 's' to save.

## 🎹 Running the Piano

```bash
python3 src/main.py
```

It runs with a GUI window if a display is attached, otherwise fully headless (logging to
stdout/journald). Both `q`/`r` keys and the visual overlay only apply in GUI mode.

### Key Features:
*   **3D Depth Triggering**: Notes are triggered when a foot enters the 50mm (5cm) "active zone" above the floor.
*   **Auto-Leveling** (GUI): Press `r` to re-sample the floor plane if the camera is bumped or the floor surface changes.
*   **Graceful shutdown**: Ctrl-C or `SIGTERM` (`systemctl stop`) stops the camera and audio cleanly.

## 🔁 Run on Boot (optional, headless)

A sample unit file is provided at `docs/floor-piano.service`. Edit the `User`, paths and
audio settings, then:
```bash
sudo cp docs/floor-piano.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now floor-piano
journalctl -u floor-piano -f      # watch the logs
```
Calibrate once (with a monitor or headless) so `src/config.json` exists before enabling the service.

## 🔧 Performance & Optimization
*   **~30 FPS**: Limited by the Astra Pro depth stream (typ. 640×480 @ 30 FPS).
*   **Latency**: Roughly 40–80 ms from foot contact to audio (one depth frame ~33 ms + pygame buffer ~6 ms).
*   **Power**: If you see "Under-voltage" warnings, the Astra Pro may disconnect. Use the **Official 27W PSU**.
