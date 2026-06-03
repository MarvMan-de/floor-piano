# 🎹 Floor Piano v2.0

Interactive floor piano system for a Raspberry Pi 5 + Orbbec Astra Pro (3D depth camera) hardware stack.

## 🚀 Features

- **24-Key Keyboard**: 2 full octaves — 14 white + 10 black keys (C4–B5), with a real
  piano layout (black keys narrower, at the back, none between E–F or B–C)
- **3D Depth Triggering**: Low-latency foot detection using Orbbec Astra Pro depth sensing
- **ArUco Calibration**: Automatic corner detection via printed markers
- **Low-Latency Audio**: PyGame audio engine with per-note samples
- **Auto-Leveling**: Floor depth sampled (median) at calibration, re-levelable live (RANSAC plane-fit planned — see VERSION_2_PLAN)

## 📁 Project Structure

```
floor-piano/
├── src/                    # Source code
│   ├── main.py            # Main piano application
│   ├── audio.py           # Audio engine
│   ├── calibrate.py       # Automatic calibration
│   └── sounds/            # Audio samples
├── docs/                  # Documentation
│   ├── SETUP.md          # Hardware setup guide
│   ├── SHOPPING_LIST.md  # Required components
│   └── VERSION_2_PLAN.md # Development roadmap
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Audio Samples** (offline, no download needed):
   ```bash
   python src/sounds/generate_samples.py
   ```

3. **Calibrate System**:
   ```bash
   python src/calibrate.py
   ```

4. **Run Piano**:
   ```bash
   python src/main.py
   ```

## 🧪 Development & Tests

The trigger logic lives in `src/detection.py` and is hardware-free, so it can be
tested without a camera:

```bash
pip install -r requirements-dev.txt
pytest
```

## 📋 Hardware Requirements

- Raspberry Pi 5 (8GB RAM)
- Orbbec Astra Pro 3D Camera
- Official Raspberry Pi 27W Power Supply
- USB Audio Adapter

## 🎯 Performance Targets

- **~30 FPS**: Real-time depth processing (Astra Pro depth-stream limit)
- **Low latency**: ~40–80 ms from foot contact to audio playback
- **Auto-Recovery**: Floor level can be re-sampled on the fly if the camera is bumped

## 🔧 Development Status

**Version 2.0** - Professional Headless Edition
- ✅ 3D Depth Triggering (pyorbbecsdk)
- ✅ ArUco Marker Calibration
- ✅ Low-Latency Audio Engine
- 🚧 Systemd Service
- 🚧 GPIO Status Indicators
- 💤 Hailo-8L Pose Integration (deferred — not part of the current hardware plan)

---

*Built for professional interactive installations and exhibitions*