#!/usr/bin/env python3
"""Generate the piano note samples as real 16-bit WAV files.

Covers a full chromatic 2-octave keyboard (C4..B5 = 24 notes) so the 14 white +
10 black floor-piano keys all have a sample. Depends only on the Python standard
library and never touches the network, so it cannot break because some external
download URL went dead.

Sharps are written with 's' in the filename (C#4 -> Cs4.wav) to keep filenames
shell-safe; this matches detection.note_filename().

Usage:
    python3 generate_samples.py
"""

import math
import os
import struct
import wave

SAMPLE_RATE = 44100
CHANNELS = 2          # matches pygame.mixer.pre_init(44100, -16, 2, 256)
DURATION = 1.2        # seconds
AMPLITUDE = 0.6       # peak before 16-bit scaling, leaves headroom

# Chromatic semitone names within an octave.
SEMITONES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

START_OCTAVE = 4
OCTAVES = 2           # C4..B5 -> 24 notes

# Relative levels of the first few harmonics — a slightly richer timbre.
HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)]


def build_note_list(start_octave=START_OCTAVE, octaves=OCTAVES):
    """Ordered chromatic note names, e.g. ['C4','C#4',...,'B5']."""
    return [f"{s}{start_octave + o}" for o in range(octaves) for s in SEMITONES]


def note_to_freq(name):
    """Equal-tempered frequency (Hz) of a note name like 'C4' or 'C#4'."""
    semitone = SEMITONES.index(name[:-1])
    octave = int(name[-1])
    midi = (octave + 1) * 12 + semitone   # C4 -> 60, A4 -> 69
    return 440.0 * 2 ** ((midi - 69) / 12)


def note_filename(name):
    """'C#4' -> 'Cs4.wav' (matches detection.note_filename)."""
    return name.replace("#", "s") + ".wav"


def render_note(freq):
    """Return a list of int16 sample values for one note (mono)."""
    n_samples = int(SAMPLE_RATE * DURATION)
    harmonic_norm = sum(level for _, level in HARMONICS)
    samples = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        envelope = math.exp(-3.0 * t)     # struck-string decay
        value = sum(level * math.sin(2.0 * math.pi * freq * mult * t)
                    for mult, level in HARMONICS)
        value = (value / harmonic_norm) * envelope * AMPLITUDE
        value = max(-1.0, min(1.0, value))
        samples.append(int(value * 32767))
    return samples


def write_wav(path, mono_samples):
    """Write mono int16 samples to a (stereo) 16-bit PCM WAV file."""
    with wave.open(path, "w") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for s in mono_samples:
            frames.extend(struct.pack("<h", s) * CHANNELS)
        wav.writeframes(bytes(frames))


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    notes = build_note_list()
    print(f"Generating {len(notes)} samples into {out_dir} ...")
    for note in notes:
        write_wav(os.path.join(out_dir, note_filename(note)), render_note(note_to_freq(note)))
        print(f"  {note_filename(note)}  ({note_to_freq(note):.2f} Hz)")
    print("Done.")


if __name__ == "__main__":
    main()
