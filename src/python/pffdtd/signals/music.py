# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np


def hz_to_note(frequency, A4=440.0):
    A4_position = 9
    note_names = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]

    semitones_from_A4 = 12 * np.log2(frequency / A4)
    semitone_offset = round(semitones_from_A4)

    octave = 4 + (A4_position + semitone_offset) // 12
    note_position = (A4_position + semitone_offset) % 12
    note_name = note_names[note_position]

    return f"{note_name}{octave}"


def midi_key_color(note: int) -> str:
    """
    Get the color of the key ("black" or "white") for a MIDI note number.

    Parameters:
        note: MIDI note number (0-127).

    Returns:
        str: "black" or "white".
    """
    assert isinstance(note, int)
    assert note >= 0
    is_black = (note % 12) in [1, 3, 6, 8, 10]
    return 'black' if is_black else 'white'
