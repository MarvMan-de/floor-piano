import pygame
import os

class PianoAudio:
    """Piano Audio engine — ultra-low latency via pygame mixer."""
    def __init__(self, samples_dir=None):
        # Default to sounds/ next to this script, not CWD
        if samples_dir is None:
            samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

        # buffer=256 → ~6ms latency at 44.1kHz
        pygame.mixer.pre_init(44100, -16, 2, 256)
        pygame.mixer.init()

        self.samples = {}
        self.active_notes = set()

        for note in ["C", "D", "E", "F", "G", "A", "B"]:
            path = os.path.join(samples_dir, f"{note}.wav")
            if os.path.exists(path):
                self.samples[note] = pygame.mixer.Sound(path)
            else:
                print(f"Warning: Sample for {note} not found at {path}")

    def update(self, current_active_keys):
        to_trigger = set(current_active_keys) - self.active_notes
        for note in to_trigger:
            if note in self.samples:
                self.samples[note].play()
        self.active_notes = set(current_active_keys)

    def close(self):
        pygame.mixer.quit()

    def cleanup(self):
        self.close()
