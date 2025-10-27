# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np
import pytest

from pffdtd.signals.music import (
    cents_deviation,
    frequency_with_cents,
    hz_to_note,
    midi_key_color,
    note_frequencies,
    note_to_midi
)


def test_cents_deviation():
    assert np.allclose(cents_deviation(440, 220), -1200, rtol=0.01)
    assert np.allclose(cents_deviation(440, 880), +1200, rtol=0.01)

    assert np.allclose(cents_deviation(440, 415.30), -100, rtol=0.01)
    assert np.allclose(cents_deviation(440, 466.16), +100, rtol=0.01)


def test_frequency_with_cents():
    # octave
    assert np.allclose(frequency_with_cents(440, -1200), 220)
    assert np.allclose(frequency_with_cents(440, +1200), 880)

    # semitone
    assert np.allclose(frequency_with_cents(440, -100), 415.30, rtol=0.01)
    assert np.allclose(frequency_with_cents(440, +100), 466.16, rtol=0.01)


def test_note_to_midi():
    assert note_to_midi('A0') == 21
    assert note_to_midi('A#0') == 22
    assert note_to_midi('Bb0') == 22
    assert note_to_midi('C4') == 60
    assert note_to_midi('A4') == 69

    with pytest.raises(ValueError, match=r"Invalid note format: 'A4c'"):
        note_to_midi('A4c')


def test_midi_key_color():
    assert midi_key_color(0) == 'white'
    assert midi_key_color(69) == 'white'
    assert midi_key_color(70) == 'black'


def test_hz_to_note():
    assert hz_to_note(55.0) == 'A1'
    assert hz_to_note(110.0) == 'A2'
    assert hz_to_note(220.0) == 'A3'
    assert hz_to_note(440.0) == 'A4'
    assert hz_to_note(880.0) == 'A5'

    assert hz_to_note(8.18) == 'C-1'
    assert hz_to_note(207.65) == 'G#3'

    assert hz_to_note(55.25, A4=442.0) == 'A1'
    assert hz_to_note(110.5, A4=442.0) == 'A2'
    assert hz_to_note(221.0, A4=442.0) == 'A3'
    assert hz_to_note(442.0, A4=442.0) == 'A4'
    assert hz_to_note(884.0, A4=442.0) == 'A5'


def test_note_frequencies():
    ref = [27.5, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.0, 51.91, 55.0]
    assert np.allclose(note_frequencies('A0', 'A1'), ref, rtol=0.01)
