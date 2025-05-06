# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

from pffdtd.signals.music import hz_to_note, midi_key_color


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
