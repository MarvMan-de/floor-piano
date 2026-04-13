# 🎹 Floor Piano v2.0

Professional-grade interactive floor piano system optimized for Raspberry Pi 5 + Hailo-8L + Orbbec Astra Pro hardware stack.

## 🚀 Features

- **3D Depth Triggering**: Ultra-low latency (<10ms) using Orbbec Astra Pro depth sensing
- **Headless Operation**: Autonomous calibration with ArUco markers - no monitor required
- **Hailo-8L AI Acceleration**: 90 FPS pose estimation and depth processing
- **Professional Audio**: Low-latency PyGame audio engine with high-quality samples
- **Auto-Leveling**: Dynamic floor plane detection with RANSAC algorithm

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

2. **Calibrate System**:
   ```bash
   python src/calibrate.py
   ```

3. **Run Piano**:
   ```bash
   python src/main.py
   ```

## 📋 Hardware Requirements

- Raspberry Pi 5 (8GB RAM)
- Raspberry Pi AI Kit (Hailo-8L + M.2 HAT+)
- Orbbec Astra Pro 3D Camera
- Official Raspberry Pi 27W Power Supply
- USB Audio Adapter

## 🎯 Performance Targets

- **90 FPS**: Real-time depth processing
- **<10ms Latency**: From foot detection to audio playback
- **Headless Operation**: Zero UI required after initial setup
- **Auto-Recovery**: Self-correcting calibration on camera movement

## 🔧 Development Status

**Version 2.0** - Professional Headless Edition
- ✅ 3D Depth Triggering
- ✅ ArUco Marker Calibration
- ✅ Low-Latency Audio Engine
- 🚧 Hailo-8L Integration
- 🚧 Systemd Service
- 🚧 GPIO Status Indicators

---

*Built for professional interactive installations and exhibitions*