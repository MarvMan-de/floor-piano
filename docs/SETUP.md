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
   pip install -r requirements.txt && pytest
   ```
   Detection (blob assignment, debounce, press-height band), geometry, keyboard
   layout, mat auto-calibration, the full FloorPiano pipeline with injected
   camera/audio, decode and config logic — 108 tests.
2. **Samples (stdlib only):**
   ```bash
   python3 src/sounds/generate_samples.py
   ```
3. **Audio device (pygame):** confirm your USB speaker works.
   ```bash
   python3 src/sounds/play_test.py    # plays all 24 chromatic notes C4..B5
   ```
4. **Full pipeline minus camera (cv2 + pygame):** a synthetic foot sweeps the keys —
   you should *hear* the scale. This proves warp → detect → audio end-to-end.
   ```bash
   python3 src/demo_mock.py
   ```
   To test against a **recorded video instead of a synthetic sweep**, feed it an
   MP4 of the mat. Use `--motion` for a normal phone clip: the foot is whatever
   differs from the median background. The key grid is **auto-calibrated from
   the mat itself** (corners, orientation — portrait/upside-down/mirrored clips
   all work — and a sub-pixel refinement onto the painted keys). Add `--show`
   for the live detection view, `--out result.mp4` to save it, `--no-audio` to
   run without pygame:
   ```bash
   python3 src/demo_video.py clip.mp4 --motion --show
   python3 src/demo_video.py clip.mp4 --motion --no-audio --out result.mp4
   ```
   (The default brightness mode — gray value → millimetres — is only for clips
   that actually encode depth; `--corners` / `--full-frame` override the
   auto-calibration if needed.)
5. **Print the markers (cv2):** prepare for on-site calibration.
   ```bash
   python3 tools/generate_aruco.py    # writes markers/marker_0..3.png
   ```

Only the Orbbec depth stream (`calibrate.py` / `main.py`) and the RGB→depth
registration need the real camera.

## 📷 Camera Placement

The Astra Pro is a structured-light depth camera — mind its physics:

*   **Minimum range ~0.6 m.** Anything closer reads as 0 (invalid). Mount high enough.
*   **Field of view.** Looking straight down from ~1.0 m covers only ~1.1 m width. The
    default layout is **24 keys (14 white + 10 black = 2 octaves, C4–B5)**, so the mat is
    wide — mount the camera high (e.g. 2.0–2.5 m) so the whole keyboard fits the frame.
*   **Lighting.** Structured-light IR is disturbed by direct sunlight; dark/glossy floors
    absorb IR and create holes (0-pixels) that read as "no foot". Prefer indoor, matte,
    light-coloured surfaces.
*   **Calibration marker convention.** The corner is taken as the **centre** of each
    ArUco marker, so place the markers' centres exactly on the 4 playable-area corners
    (the playable zone ends up inset by ~half a marker — this is expected).

## 📐 Automatic Calibration (ArUco or the mat itself)

Calibration runs in a GUI window (press `s` to save) or, if no display is attached,
**headless** — it auto-saves once detection stays stable for a short period (and gives
up with an error after 2 minutes instead of hanging). The floor depth is the median
over the stable window, with a spread check so someone walking through the scene
can't poison it. **Keep the mat clear of feet while calibrating.**

Two corner sources (`--source auto|aruco|mat`, default `auto` = ArUco first):

*   **ArUco markers** — print 4 markers from the `DICT_4X4_50` dictionary
    (IDs 0=Top-Left, 1=Top-Right, 2=Bottom-Right, 3=Bottom-Left) and place their
    *centres* on the playable-area corners.
*   **The printed mat itself (no markers needed)** — the mat is detected as the big
    bright rectangle, its orientation is resolved from the black-key pattern, and the
    grid is refined onto the painted keys. This is the fallback whenever the markers
    aren't found, so on site you can simply run:
    ```bash
    python3 src/calibrate.py            # markers optional
    python3 src/calibrate.py --source mat   # skip ArUco entirely
    ```

## 🎹 Running the Piano

```bash
python3 src/main.py
```

It runs with a GUI window if a display is attached, otherwise fully headless (logging to
stdout/journald). Both `q`/`r` keys and the visual overlay only apply in GUI mode.

### Key Features:
*   **3D Depth Triggering**: a foot in the 50mm "active zone" above the floor triggers —
    but only within 250mm of the floor (`max_press_height`), so a foot swinging
    mid-step or a knee over the mat stays silent.
*   **One foot = one key**: the above-floor pixels are split into blobs; each blob
    presses exactly the key it covers most (with boundary hysteresis), so stepping on
    a key edge can't fire two notes — while two feet still play chords.
*   **Debounced release**: a key releases only after 3 consecutive empty frames, so
    depth noise can't machine-gun a held note.
*   **Auto-Leveling** (GUI): Press `r` to re-sample the floor plane if the camera is
    bumped (people standing on the mat are masked out of the sample).
*   **Watchdog**: if the camera stops delivering frames for ~5s the service exits with
    an error so systemd restarts it.
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
