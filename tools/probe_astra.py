#!/usr/bin/env python3
"""Phase-0 hardware probe for the Orbbec Astra Pro on the Raspberry Pi 5.

Run this the moment the camera is plugged in — BEFORE touching anything else.
It answers the single question the whole project hinges on: does the Astra Pro
deliver a depth frame on this Pi via OpenNI2?

    python3 tools/probe_astra.py

It is read-only and safe: it lists the USB device, tries to open the OpenNI2
depth stream, grabs one frame, and prints a clear PASS / FAIL with the next
step. Depth is the hard part (OpenNI2); RGB is an ordinary UVC webcam and is
checked only as a bonus. See docs/ASTRA_PRO_PI5_SETUP.md for the setup that has
to be in place first (OpenNI2 SDK + `pip install openni`).
"""

import subprocess
import sys

ORBBEC_VID = "2bc5"  # Orbbec USB vendor id


def hr(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_usb():
    hr("1. USB — is the camera enumerated?")
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        print(f"  ?  could not run lsusb ({e}); skipping this check.")
        return None
    hits = [ln for ln in out.splitlines() if ORBBEC_VID in ln.lower()]
    if hits:
        print("  OK  Orbbec device(s) found:")
        for ln in hits:
            print(f"        {ln}")
        print("      (expect TWO nodes: the depth sensor + a separate UVC RGB cam)")
        return True
    print(f"  FAIL  no USB device with vendor id {ORBBEC_VID} found.")
    print("        -> check the cable/port (use a USB-A port, not a hub), and")
    print("           that the camera has power. Re-run `lsusb` after replugging.")
    return False


def check_depth():
    hr("2. DEPTH — can OpenNI2 open the depth stream? (the decisive test)")
    try:
        from openni import openni2
    except ImportError:
        print("  FAIL  the `openni` Python bindings are not installed.")
        print("        -> pip install openni   AND install the Orbbec OpenNI2 SDK")
        print("           redist (set OPENNI2_REDIST). See docs/ASTRA_PRO_PI5_SETUP.md.")
        return False

    import os
    redist = os.environ.get("OPENNI2_REDIST") or os.environ.get("OPENNI2_REDIST64")
    try:
        openni2.initialize(redist) if redist else openni2.initialize()
    except Exception as e:
        print(f"  FAIL  openni2.initialize() failed: {e}")
        print("        -> OpenNI2 can't find libOpenNI2.so / the Orbbec driver.")
        print("           Set OPENNI2_REDIST to the SDK redist dir (the folder with")
        print("           libOpenNI2.so + OpenNI2/Drivers/). See the SETUP doc.")
        return False

    try:
        dev = openni2.Device.open_any()
        stream = dev.create_depth_stream()
        stream.start()
        frame = stream.read_frame()
        h = frame.height if hasattr(frame, "height") else frame.get_height()
        w = frame.width if hasattr(frame, "width") else frame.get_width()
        buf = frame.get_buffer_as_uint16()
        try:
            import numpy as np
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(h, w)
            mid = int(arr[h // 2, w // 2])
            valid = int((arr > 0).sum())
            extra = f", centre={mid}mm, valid pixels={valid}/{h * w}"
        except Exception:
            extra = ""
        stream.stop()
        print(f"  PASS  got a depth frame: {w}x{h}{extra}")
        print("        -> the hard part works. Set FLOOR_PIANO_CAMERA=openni2 and run main.py.")
        return True
    except Exception as e:
        print(f"  FAIL  could not read a depth frame: {e}")
        msg = str(e).lower()
        if "endpoint" in msg or "usb" in msg or "transfer" in msg:
            print("        -> looks like the known ARM64 'USB endpoint not found' issue.")
            print("           Try OpenNI2 SDK 2.3.0.63 (the last known-good), a different")
            print("           OS image, or escalate to the Hochschule. See the SETUP doc")
            print("           troubleshooting section.")
        return False
    finally:
        try:
            openni2.unload()
        except Exception:
            pass


def check_rgb():
    hr("3. RGB — bonus: is the UVC colour cam visible? (not required for triggering)")
    try:
        import cv2
    except ImportError:
        print("  ?  opencv not importable here; skipping.")
        return
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            print(f"  OK  a camera responds at /dev/video{idx} (likely the Astra Pro RGB).")
            return
    print("  ?  no UVC camera responded at video0..3 (check `v4l2-ctl --list-devices`).")


def main():
    print("Astra Pro / Raspberry Pi 5 — Phase-0 probe")
    usb = check_usb()
    depth_ok = check_depth()
    check_rgb()

    hr("VERDICT")
    if depth_ok:
        print("  GREEN — depth works. The OpenNI2 backend is good to go:")
        print("          FLOOR_PIANO_CAMERA=openni2 python3 src/main.py")
        return 0
    print("  RED — no depth yet. Depth is the blocker, not the app.")
    if usb is False:
        print("        Start at step 1: the camera isn't even on the USB bus.")
    else:
        print("        Work through docs/ASTRA_PRO_PI5_SETUP.md (OpenNI2 install +")
        print("        the troubleshooting section) and re-run this probe.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
