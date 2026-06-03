#!/usr/bin/env python3
"""Generate the 7 piano note samples (C4-B4) as real 16-bit WAV files.

This is the reliable, offline way to get playable samples: it depends only on
the Python standard library and never touches the network, so it cannot break
because some external download URL went dead (which is exactly what happened to
the committed .wav files — they were HTML 404 pages, not audio).

The tones are synthesised (fundamental + a few harmonics + a percussive decay
envelope). They sound piano-ish rather than like a real grand; swap in proper
samples via download_samples.sh if you have a good source.

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

# Equal-tempered 4th-octave frequencies (Hz).
NOTE_FREQS = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
    "A": 440.00,
    "B": 493.88,
}

# Relative levels of the first few harmonics — gives a slightly richer timbre.
HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)]


def render_note(freq):
    """Return a list of int16 sample values for one note (mono)."""
    n_samples = int(SAMPLE_RATE * DURATION)
    samples = []
    harmonic_norm = sum(level for _, level in HARMONICS)
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Exponential decay so the note rings out like a struck string.
        envelope = math.exp(-3.0 * t)
        value = 0.0
        for mult, level in HARMONICS:
            value += level * math.sin(2.0 * math.pi * freq * mult * t)
        value = (value / harmonic_norm) * envelope * AMPLITUDE
        # Clamp and scale to signed 16-bit.
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
            packed = struct.pack("<h", s)
            frames.extend(packed * CHANNELS)  # same signal to every channel
        wav.writeframes(bytes(frames))


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Generating {len(NOTE_FREQS)} samples into {out_dir} ...")
    for note, freq in NOTE_FREQS.items():
        path = os.path.join(out_dir, f"{note}.wav")
        write_wav(path, render_note(freq))
        print(f"  {note}.wav  ({freq:.2f} Hz)")
    print("Done.")


if __name__ == "__main__":
    main()
