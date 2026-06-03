import logging
import os

import pygame

from detection import keyboard_note_names, newly_pressed, note_filename

log = logging.getLogger(__name__)


class PianoAudio:
    """Piano audio engine — low latency via the pygame mixer.

    Loading a sample is best-effort: a missing or corrupt .wav is logged and
    skipped instead of crashing the whole application. The corresponding key
    simply stays silent until a valid sample is provided.
    """

    def __init__(self, samples_dir=None, keys=None):
        # Default to sounds/ next to this script, not the current working dir.
        if samples_dir is None:
            samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
        self.keys = keys if keys else keyboard_note_names()
        self.samples = {}
        self.active_notes = set()
        self._mixer_ok = False

        # buffer=256 -> ~6ms latency at 44.1kHz. If no audio device is available
        # (common headless, or USB audio not ready yet), run silently rather than
        # crashing — important under systemd Restart=on-failure.
        try:
            pygame.mixer.pre_init(44100, -16, 2, 256)
            pygame.mixer.init()
            self._mixer_ok = True
        except pygame.error as e:
            log.warning("Audio mixer unavailable (%s) — running without sound.", e)
            return

        for note in self.keys:
            path = os.path.join(samples_dir, note_filename(note))
            if not os.path.exists(path):
                log.warning("Sample for '%s' not found at %s — key will be silent.", note, path)
                continue
            try:
                self.samples[note] = pygame.mixer.Sound(path)
            except (pygame.error, FileNotFoundError) as e:
                # e.g. an HTML error page saved as .wav, or a truncated file.
                log.warning("Could not load sample for '%s' (%s): %s — key will be silent.",
                            note, path, e)

        if not self.samples:
            log.warning("No valid audio samples loaded. Run sounds/generate_samples.py "
                        "(or sounds/download_samples.sh) to create them.")

    def update(self, current_active_keys):
        """Play the notes that became active since the previous frame (edge-triggered)."""
        if self._mixer_ok:
            for note in newly_pressed(current_active_keys, self.active_notes):
                sound = self.samples.get(note)
                if sound is not None:
                    sound.play()
        self.active_notes = set(current_active_keys)

    def close(self):
        if self._mixer_ok:
            pygame.mixer.quit()
