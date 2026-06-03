#!/usr/bin/env python3
"""Play the 7 note samples in order — a quick USB-audio smoke test (no camera).

Run:
    python3 src/sounds/play_test.py

Confirms pygame + your audio device (e.g. USB speaker) work. Run
generate_samples.py first if the .wav files are missing. If you hear nothing,
the problem is the audio device/ALSA, not the piano logic.
"""

import os
import sys
import time

import pygame

NOTES = ["C", "D", "E", "F", "G", "A", "B"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pygame.mixer.pre_init(44100, -16, 2, 256)
    pygame.mixer.init()
    print("Audio device:", pygame.mixer.get_init())

    played = 0
    for note in NOTES:
        path = os.path.join(here, f"{note}.wav")
        if not os.path.exists(path):
            print(f"  missing {path} — run generate_samples.py first")
            continue
        print("  playing", note)
        pygame.mixer.Sound(path).play()
        played += 1
        time.sleep(0.6)

    time.sleep(0.6)
    pygame.mixer.quit()
    print(f"Done — played {played}/{len(NOTES)} notes.")
    if played == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
