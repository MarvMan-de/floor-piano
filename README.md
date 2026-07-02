# 🎹 Floor Piano

Interactive floor piano: an **Orbbec Gemini 335** depth camera watches a printed
piano mat (or, while testing, a tablet showing the keys); stepping/pressing on a
key plays its note. Runs on a **Raspberry Pi 5**, configured and played through a
**web UI** reachable over the Pi's hotspot.

## 🚀 How it works

1. **Web UI** (the heart of the project): shows the live camera image, you place
   the mat's 4 corners and the 24 keys are perspective-projected onto them.
2. **Capture surface**: one click stores the empty surface as a per-pixel depth
   reference — works even when the surface is tilted.
3. **Play**: a press is a finger/foot-sized *blob* of pixels right **at** the
   surface (thin contact band above the reference). A hand or body passing over
   the keys is higher and stays silent. Triggered notes flash in the UI and play
   in the browser.

## 🛠 Quick Start

```bash
# with uv (recommended)
uv sync
uv run python -m webui.server --port 8000

# or with plain pip
pip install -r requirements.txt
python -m webui.server --port 8000
```

Open **http://localhost:8000** (or `http://<pi-ip>:8000` from a phone on the
Pi's hotspot), then:

1. **Mat-Ecken setzen** — click the mat's 4 corners → 24 keys appear (drag the
   corner handles to fine-tune; the keys follow, perspective-correct).
2. **Oberfläche erfassen** — with no fingers/feet in view.
3. **Play** — press the keys; the browser plays the notes.

First time with the camera on Linux: install the udev rules once, then re-plug:

```bash
sudo sh .venv/lib/python*/site-packages/pyorbbecsdk/shared/install_udev_rules.sh
```

## 📁 Project Structure

```
floor-piano/
├── webui/                  # The web app (FastAPI + vanilla JS) — config + play
│   ├── server.py          # API: tiles, corners, surface capture, play mode, MJPEG
│   ├── camera_source.py   # Gemini colour + D2C-aligned depth (pyorbbecsdk)
│   ├── depth_detect.py    # contact-band + blob press detection
│   └── static/            # index.html, app.js, style.css
├── src/                    # Standalone depth pipeline + shared logic
│   ├── main.py            # Headless piano (keyboard-grid path, pygame audio)
│   ├── detection.py       # Hardware-free trigger logic (unit-tested)
│   ├── calibrate.py       # ArUco / printed-mat calibration for the src path
│   ├── mat_calibration.py # Marker-less mat detection (corners from painted keys)
│   ├── placement.py       # Camera-placement coach (structured hints)
│   └── sounds/            # Note samples (C4–B5, generate_samples.py)
├── tools/                  # probe_gemini.py / view_gemini.py (live depth checks)
├── tests/                  # pytest suite (hardware-free)
└── docs/                   # Setup guides, project documentation, code review
```

## 🧪 Tests

All detection logic is hardware-free and covered by the suite:

```bash
uv run pytest        # or: pytest
```

## 📋 Hardware

- Raspberry Pi 5 (8 GB — far more than needed; detection costs ~10 ms/frame)
- **Orbbec Gemini 335** (USB 3.0; depth FOV 90°×65°, so a 2 m mat fits from
  ~1.1–1.2 m mounting height)
- USB audio adapter / speakers (for standalone `src/main.py` playback)

Legacy note: an original Orbbec Astra (OpenNI2) is still supported as a fallback
via `FLOOR_PIANO_CAMERA=openni2` — see `docs/ASTRA_PI5_SETUP.md`.

## 🔧 Status

- ✅ Gemini 335 live at 30 FPS (format/scale verified with `tools/probe_gemini.py`)
- ✅ Web UI: corner-based key placement, click-to-play, depth Play mode
- ✅ Touch = contact band + blob detection (hover/pass-over stays silent)
- 🚧 Threshold tuning on the real mat (`webui/depth_detect.py` constants)
- 🚧 Pi hotspot deployment (systemd service for `webui.server`)

Full development log: `docs/PROJEKTDOKUMENTATION.md` · review: `docs/CODE_REVIEW.md`
