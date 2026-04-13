# 🎹 University Floor Piano: Professional Shopping List

This list is specifically optimized for your **Raspberry Pi 5 + Hailo-8L + Orbbec Astra Pro** hardware stack. This setup provides 90 FPS AI pose tracking combined with hardware-level 3D depth sensing.

### 1. Core Processing & AI
*   **Raspberry Pi 5 (8GB RAM)**: The 8GB version is required for handling the high-bandwidth 3D depth stream and the Hailo AI pipeline simultaneously.
*   **Raspberry Pi AI Kit**: Includes the **Hailo-8L M.2 AI Module** (13 TOPS) and the **M.2 HAT+**. This is what gives you the "pro" performance.
*   **Official Raspberry Pi Active Cooler**: Non-negotiable. The Pi 5 and the AI module generate significant heat during real-time 90 FPS inference.

### 2. The "Eyes" (3D Depth & Vision)
*   **Orbbec Astra Pro (3D/Depth/RGB Camera)**: **MANDATORY.** This camera provides the 3D floor-plane data needed to eliminate false triggers.
*   **High-Quality USB 3.0 Cable (USB-A to USB-B/Micro/C depending on your specific Astra model)**: Ensure it is a "SuperSpeed" rated cable. The Astra requires the full bandwidth of the Pi 5's blue USB ports.
*   **3.0m - 5.0m USB 3.0 Extension (Active)**: If you are mounting the camera on a high ceiling or tall tripod to cover a large floor area, you will need an **active** USB 3.0 extension cable to prevent signal drop-outs.

### 3. Power & Audio (Critical for Pi 5)
*   **Official Raspberry Pi 27W USB-C Power Supply**: **CRITICAL.** The Pi 5 + Hailo + Astra (USB powered) draw more than 15W. Without the 27W supply, the camera will likely disconnect or the Pi will throttle.
*   **USB to 3.5mm Audio Adapter** OR **USB Powered Speakers**: The Pi 5 has no analog audio jack. You need a USB solution to hear the piano notes.
*   **3.5mm Male-to-Male Cable**: To connect your audio adapter to the university's PA system or large speakers.

### 4. Setup & Installation
*   **MicroSD Card (64GB+ Class 10/UHS-I)**: Sufficient space for the OS, Hailo firmware, and recorded performance logs.
*   **Micro-HDMI to HDMI Cable**: To connect the Pi 5 to a monitor for initial setup and calibration.
*   **Heavy Duty Tripod or C-Clamp Mount**: To position the Astra Pro securely looking down at the floor.

### 5. Floor Materials
*   **Gaffer Tape (High Visibility)**: To mark the "keys" on the floor so performers know where to step.
*   **Anti-Fatigue Floor Mats (Optional)**: If this is for a long-duration exhibition, placing these under the camera's view makes it more comfortable for users.

---
### 🛒 Quick Checkout Checklist:
- [ ] Raspberry Pi 5 (8GB)
- [ ] Raspberry Pi AI Kit (Hailo-8L + M.2 HAT+)
- [ ] Raspberry Pi Active Cooler
- [ ] Orbbec Astra Pro (USB 3.0 version)
- [ ] Official 27W Power Supply (Black or White)
- [ ] USB 3.0 Extension Cable (Active, if mounting high)
- [ ] USB Audio Adapter / USB Speakers
- [ ] 64GB MicroSD Card
- [ ] Micro-HDMI to HDMI Cable
- [ ] Gaffer Tape (for floor markers)
