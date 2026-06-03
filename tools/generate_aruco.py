#!/usr/bin/env python3
"""Generate the 4 ArUco markers (DICT_4X4_50, IDs 0-3) as PNGs to print.

Run:
    python3 tools/generate_aruco.py [output_dir] [pixels]

Print them and place at the mat corners: 0=Top-Left, 1=Top-Right, 2=Bottom-Right,
3=Bottom-Left. Print as large as practical (better detection); a quiet white border
around each marker helps a lot. Needs opencv-contrib (cv2.aruco).
"""

import os
import sys

import cv2


def main():
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "markers")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    os.makedirs(out_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid in (0, 1, 2, 3):
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, size)
        path = os.path.join(out_dir, f"marker_{mid}.png")
        cv2.imwrite(path, img)
        print(f"wrote {path}")

    print("Print these. Placement: 0=TL, 1=TR, 2=BR, 3=BL.")


if __name__ == "__main__":
    main()
