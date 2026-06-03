# 🎹 University Floor Piano: Professional Shopping List

This list targets a **Raspberry Pi 5 + Orbbec Astra Pro** hardware stack using 3D depth sensing for foot detection.

### 1. Core Processing
*   **Raspberry Pi 5 (8GB RAM)**: Comfortably handles the 3D depth stream and the trigger logic.
*   **Official Raspberry Pi Active Cooler**: Recommended for sustained operation, though the depth-only workload is light on the CPU.

> ℹ️ **Note:** The original plan included a Raspberry Pi AI Kit (Hailo-8L NPU). The current depth-based piano does **not** use it — it has been dropped for now. Add it back only if you later want AI pose estimation (multi-person tracking).

### 2. The "Eyes" (3D Depth & Vision)
*   **Orbbec Astra Pro (3D/Depth/RGB Camera)**: **MANDATORY.** This camera provides the 3D floor-plane data needed to eliminate false triggers.
*   **Quality USB Cable (USB-A to USB-B/Micro/C depending on your specific Astra model)**: The original Astra Pro is a **USB 2.0** device, so a normal shielded USB 2.0 cable is sufficient — there is no USB 3.0 / "SuperSpeed" requirement.
*   **3.0m - 5.0m Active USB Extension**: If you mount the camera on a high ceiling or tall tripod to cover a large floor area, use an **active** (powered) extension to prevent signal drop-outs.

### 3. Power & Audio (Critical for Pi 5)
*   **Official Raspberry Pi 27W USB-C Power Supply**: **CRITICAL.** The Pi 5 plus a USB-powered Astra can exceed the Pi's default USB current budget. Without the 27W supply, the camera may disconnect or the Pi will throttle.
*   **USB to 3.5mm Audio Adapter** OR **USB Powered Speakers**: The Pi 5 has no analog audio jack. You need a USB solution to hear the piano notes.
*   **3.5mm Male-to-Male Cable**: To connect your audio adapter to the university's PA system or large speakers.

### 4. Setup & Installation
*   **MicroSD Card (64GB+ Class 10/UHS-I)**: Sufficient space for the OS, dependencies, and recorded performance logs.
*   **Micro-HDMI to HDMI Cable**: To connect the Pi 5 to a monitor for initial setup and calibration.
*   **Heavy Duty Tripod or C-Clamp Mount**: To position the Astra Pro securely looking down at the floor.

### 5. Floor Materials
*   **Gaffer Tape (High Visibility)**: To mark the "keys" on the floor so performers know where to step.
*   **Anti-Fatigue Floor Mats (Optional)**: If this is for a long-duration exhibition, placing these under the camera's view makes it more comfortable for users.

---
### 🛒 Quick Checkout Checklist:
- [ ] Raspberry Pi 5 (8GB)
- [ ] Raspberry Pi Active Cooler
- [ ] Orbbec Astra Pro (USB 3.0 version)
- [ ] Official 27W Power Supply (Black or White)
- [ ] USB 3.0 Extension Cable (Active, if mounting high)
- [ ] USB Audio Adapter / USB Speakers
- [ ] 64GB MicroSD Card
- [ ] Micro-HDMI to HDMI Cable
- [ ] Gaffer Tape (for floor markers)
