"""Tests for the offline sample generator.

These guard against the original bug (committed .wav files were actually HTML 404
pages). They use only the stdlib `wave` module, so they need no audio device.
"""
import os
import wave

import generate_samples as g


def test_render_note_has_expected_length_and_is_not_silent():
    samples = g.render_note(440.0)
    assert len(samples) == int(g.SAMPLE_RATE * g.DURATION)
    assert max(abs(s) for s in samples) > 0


def test_render_note_stays_in_int16_range():
    samples = g.render_note(261.63)
    assert all(-32768 <= s <= 32767 for s in samples)


def test_write_wav_produces_a_real_riff_wave(tmp_path):
    path = os.path.join(tmp_path, "C.wav")
    g.write_wav(path, g.render_note(261.63))

    # Header check: this is the exact thing the broken downloads failed.
    with open(path, "rb") as f:
        head = f.read(12)
    assert head[:4] == b"RIFF"
    assert head[8:12] == b"WAVE"

    with wave.open(path) as w:
        assert w.getnchannels() == g.CHANNELS
        assert w.getframerate() == g.SAMPLE_RATE
        assert w.getsampwidth() == 2  # 16-bit
        assert w.getnframes() == int(g.SAMPLE_RATE * g.DURATION)


def test_all_seven_notes_are_defined():
    assert set(g.NOTE_FREQS) == {"C", "D", "E", "F", "G", "A", "B"}
