# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, windows
from scipy.io import wavfile

from pffdtd.geometry.math import iceil
from pffdtd.common.plot import plot_styles
from pffdtd.common.wavfile import collect_wav_files


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def hz_to_note(frequency):
    # Reference frequency for A4
    A4_frequency = 440.0
    # Reference position for A4 in the note list
    A4_position = 9
    # List of note names
    note_names = ['C', 'C#', 'D', 'D#', 'E',
                  'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Calculate the number of semitones between the given frequency and A4
    semitones_from_A4 = 12 * np.log2(frequency / A4_frequency)
    # Round to the nearest semitone
    semitone_offset = round(semitones_from_A4)

    # Calculate the octave
    octave = 4 + (A4_position + semitone_offset) // 12
    # Calculate the note position
    note_position = (A4_position + semitone_offset) % 12

    # Get the note name
    note_name = note_names[note_position]

    return f"{note_name}{octave}"


def room_mode(L, W, H, m, n, p, c=343):
    return c*0.5*np.sqrt((m/L)**2 + (n/W)**2 + (p/H)**2)


def room_mode_kind(m, n, p):
    non_zero = (m != 0) + (n != 0) + (p != 0)
    if non_zero == 1:
        return 'axial'
    if non_zero == 2:
        return 'tangential'
    return 'oblique'


def room_modes(L, W, H, max_order=6, c=343):
    modes = []
    for m in range(max_order+1):
        for n in range(max_order+1):
            for p in range(max_order+1):
                if m+n+p > 0:
                    modes.append({
                        'm': m,
                        'n': n,
                        'p': p,
                        'frequency': room_mode(L, W, H, m, n, p, c=c),
                        'kind': room_mode_kind(m, n, p)
                    })

    return sorted(modes, key=lambda x: x['frequency'])


def frequency_spacing_index(modes):
    psi = 0.0
    num = len(modes)
    f0 = modes[0]['frequency']
    fn = modes[-1]['frequency']
    delta_hat = (fn-f0)/(num-1)
    for n in range(1, num):
        prev = modes[n-1]['frequency']
        mode = modes[n]
        freq = mode['frequency']
        delta = freq-prev
        psi += (delta/delta_hat)**2
    return psi / (num-1)


def detect_room_modes(
    filename,
    sim_dir,
    fmin,
    fmax,
    width,
    length,
    height,
    num_modes,
    plot,
):
    directory = sim_dir
    paths = filename
    if not paths:
        paths = collect_wav_files(directory, '*_out_normalised.wav')

    constants = h5py.File(sim_dir / 'constants.h5', 'r')
    c = float(constants['c'][...])

    L = length
    W = width
    H = height

    A = L*W
    V = A*H
    S = 2*(L*W+L*H+W*H)

    max_order = 8
    modes = room_modes(L, W, H, max_order=max_order, c=c)[:25]

    print(f"{L=:.3f}m {W=:.3f}m {H=:.3f}m")
    print(f"{A=:.2f}m^2 {S=:.2f}m^2 {V=:.2f}m^3")
    print(f"w/h={W/H:.2f} l/h={L/H:.2f} l/w={L/W:.2f}")
    print(f"FSI({len(modes)}): {frequency_spacing_index(modes):.2f}")
    print('')

    for mode in modes[:10]:
        m = mode['m']
        n = mode['n']
        p = mode['p']
        freq = mode['frequency']
        note = hz_to_note(freq)
        kind = room_mode_kind(m, n, p)
        print(f"[{m},{n},{p}] = {freq:.2f}Hz ({note}) {kind}")

    plt.rcParams.update(plot_styles)

    for file in paths:
        file = pathlib.Path(file)
        fs, buf = wavfile.read(file)
        fmin = fmin if fmin > 0 else 1
        fmax = fmax if fmax > 0 else fs/2

        window = windows.hann(buf.shape[0])
        buf *= window

        nfft = (2**iceil(np.log2(buf.shape[0])))*8
        spectrum = np.fft.rfft(buf, nfft)
        freqs = np.fft.rfftfreq(nfft, 1/fs)

        dB = 20*np.log10(np.abs(spectrum)+np.spacing(1))
        dB -= np.max(dB)
        dB += 75.0

        dB_max = np.max(dB)
        peaks, _ = find_peaks(dB, width=2, height=50.0)
        measured_mode_freqs = freqs[peaks]

        print(measured_mode_freqs[:10])

        plt.plot(freqs, dB, linestyle='-', label=f'{file.stem[:4]}')

    if num_modes > 0:
        calculated_mode_freqs = [mode['frequency']
                                 for mode in modes][:num_modes]
        plt.vlines(calculated_mode_freqs, dB_max-80, dB_max+10,
                   colors='#AAAAAA', linestyles='--', label='Modes')
        if len(paths) == 1:
            plt.plot(measured_mode_freqs, dB[peaks], 'r.',
                     markersize=10, label='Peaks')

        errors = []
        for mode in calculated_mode_freqs:
            nearest = find_nearest(measured_mode_freqs, mode)
            error = mode-nearest
            error_pct = np.abs(error)/mode*100
            errors.append({
                'Mode Hz': round(mode, 3),
                'Nearest Peak Hz': round(nearest, 3),
                'Error Hz': round(error, 3),
                'Error %': round(error_pct, 3),
            })

        print(pd.DataFrame.from_records(errors).to_markdown(index=False))

    plt.title('')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xscale('log')
    plt.ylim((dB_max-80, dB_max+5))
    plt.xlim((fmin, fmax))
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()

    if plot:
        plt.show()

    return calculated_mode_freqs, measured_mode_freqs


@click.command(name='room-modes', help='Plot room modes.')
@click.argument('filename', nargs=-1, type=click.Path(exists=True))
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--fmin', default=1.0, type=float)
@click.option('--fmax', default=200.0, type=float)
@click.option('--width', default=2.0, type=float)
@click.option('--length', default=3.0, type=float)
@click.option('--height', default=4.0, type=float)
@click.option('--num_modes', default=10, type=int)
@click.option('--plot/--no-plot', default=True)
def main(
    filename,
    sim_dir,
    fmin,
    fmax,
    width,
    length,
    height,
    num_modes,
    plot,
):
    detect_room_modes(filename, sim_dir, fmin, fmax, width,
                      length, height, num_modes, plot)
