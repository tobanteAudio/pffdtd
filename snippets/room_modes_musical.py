# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
"""
Code for paper: Searching the musical rehearsal room (Jens Holger Rindel)

- https://odeon.dk/pdf/Rindel_1_BNAM2020.pdf
"""
import matplotlib.pyplot as plt
import pandas as pd

from pffdtd.analysis.room_modes import room_modes
from pffdtd.common.plot import plot_styles
from pffdtd.signals.music import hz_to_note, note_frequencies, frequency_with_cents


def main():
    c = 343.3

    L = 6.36
    W = 4.44
    H = 3.0

    L = 6.0
    W = 3.65
    H = 3.12

    A = L*W
    V = A*H
    print(f"L = {L:.2f}m, W = {W:.2f}m, H = {H:.2f}m")
    print(f"A = {A:.2f}m^2")
    print(f"V = {V:.2f}m^3")

    modes = pd.DataFrame.from_records(room_modes(L, W, H, max_order=10, c=c))
    print(modes)

    notes = note_frequencies('A0', 'A3')
    low = frequency_with_cents(notes, -50)
    high = frequency_with_cents(notes, +50)
    notes_df = pd.DataFrame.from_dict({'center': notes, 'low': low, 'high': high})

    counts = []
    names = []
    for _, row in notes_df.iterrows():
        support = modes[(modes['frequency'] >= row['low']) & (modes['frequency'] < row['high'])]['frequency']
        counts.append(support.count())
        names.append(hz_to_note(row['center']))

    notes_df['support'] = counts
    notes_df['note'] = names

    plt.rcParams.update(plot_styles)
    plt.bar(notes_df['note'], notes_df['support'])
    plt.xlabel('Note')
    plt.ylabel('Number of room modes per semitone')
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    main()
