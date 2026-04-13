import pygame
import os

class PianoAudio:
    """Unified Piano Audio for both Standard and Pro versions."""
    def __init__(self, samples_dir="sounds"):
        # Initialize mixer with ultra low latency settings
        # buffer=256 reduces latency to ~6ms
        pygame.mixer.pre_init(44100, -16, 2, 256)
        pygame.mixer.init()
        
        self.samples = {}
        self.active_notes = set()
        
        notes = ["C", "D", "E", "F", "G", "A", "B"]
        for note in notes:
            path = os.path.join(samples_dir, f"{note}.wav")
            if os.path.exists(path):
                self.samples[note] = pygame.mixer.Sound(path)
            else:
                print(f"Warning: Sample for {note} not found at {path}")

    def update(self, current_active_keys):
        # notes that are newly pressed
        to_trigger = set(current_active_keys) - self.active_notes
        
        for note in to_trigger:
            if note in self.samples:
                self.samples[note].play()
        
        # update state for next frame
        self.active_notes = set(current_active_keys)

    def close(self):
        pygame.mixer.quit()

    def cleanup(self):
        self.close()
