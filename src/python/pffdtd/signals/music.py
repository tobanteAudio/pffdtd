# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import re
import numpy as np

_NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')


def note_to_midi(note: str) -> int:
    """
    Convert a note like 'C4', 'F#3', 'Bb2' to a MIDI number.
    MIDI 69 is A4. MIDI 60 is C4. Supports flats/sharps, negative octaves.
    """
    m = _NOTE_RE.match(note.strip())
    if not m:
        raise ValueError(f"Invalid note format: {note!r}")
    letter, accidental, octave_str = m.groups()
    letter = letter.upper()
    octave = int(octave_str)

    semitone = _NOTE_BASE[letter]
    if accidental == '#':
        semitone += 1
    elif accidental == 'b':
        semitone -= 1

    # MIDI numbers: C-1 == 0, so C(octave) = (octave + 1) * 12
    midi = (octave + 1) * 12 + semitone
    return midi


def midi_to_freq(midi: int, ref: float = 440.0) -> float:
    """Equal temperament: A4 (MIDI 69) = ref Hz."""
    return float(ref) * (2.0 ** ((midi - 69) / 12.0))


def note_frequencies(start_note: str, end_note: str, ref: float = 440.0) -> np.ndarray:
    """
    Return an array of frequencies (Hz) for all chromatic notes from start_note to end_note (inclusive).
    Notes are parsed like 'A0', 'C#4', 'Bb3'. The reference tuning is A4 = ref (Hz).
    """
    start_midi = note_to_midi(start_note)
    end_midi = note_to_midi(end_note)
    assert start_midi <= end_midi

    midis = np.arange(start_midi, end_midi + 1, dtype=int)
    freqs = np.array([midi_to_freq(m, ref=ref) for m in midis], dtype=float)
    return freqs


def frequency_with_cents(reference_freq, cents):
    """Return frequency shifted by `cents` relative to reference_freq."""
    return reference_freq * (2 ** (cents / 1200.0))


def cents_deviation(reference, measured):
    return 1200 * np.log2(measured / reference)


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
    is_black = note % 12 in [1, 3, 6, 8, 10]
    return 'black' if is_black else 'white'
