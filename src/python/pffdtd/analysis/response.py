# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.io import wavfile

from pffdtd.geometry.math import iceil
from pffdtd.signals.music import midi_key_color
from pffdtd.signals.octave import center_frequencies, octave_smoothing
from pffdtd.signals.wavfile import wavread


def plot_musical_response(
    file,
    *,
    ax=None,
    fmin=20,
    fmax=20000.0,
    fraction=12,
    pitch_ref=440.0,
    key_colors=True,
    verbose=False,
):
    fs, buf = wavread(file)
    fmax = fmax if fmax != 0.0 else fs/2
    nfft = buf.shape[-1]
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    spectrum = np.fft.rfft(buf, nfft)

    dB = 20*np.log10(np.maximum(np.abs(spectrum), 1e-9))
    dB -= np.max(dB)

    all_centre = center_frequencies(fraction, pitch_ref, 6, 6)
    centre = all_centre[(all_centre >= fmin) & (all_centre <= fmax)]
    smoothed = np.zeros_like(centre)

    for i in range(centre.shape[-1]):
        fc = centre[i]
        fl = fc / 2**(1/(2*fraction))
        fu = fc * 2**(1/(2*fraction))
        idx_l = np.searchsorted(freqs, fl, side='left')
        idx_u = np.searchsorted(freqs, fu, side='right')
        if idx_u >= idx_l:
            smoothed[i] = np.mean((dB[idx_l:idx_u+1]+85))

    if verbose:
        print(centre)
        print(smoothed)

    if not ax:
        ax = plt.gca()

    note_numbers = np.arange(centre.shape[-1])+16

    if key_colors:
        note_colors = [midi_key_color(int(n)) for n in note_numbers]
        note_colors = ['blue' if color == 'white' else color for color in note_colors]
        ax.bar(note_numbers, smoothed, facecolor=note_colors)
    else:
        ax.bar(note_numbers, smoothed)

    ax.set_xlabel('Note Number [MIDI]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(48, 92)
    ax.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)


def plot_response_compare(
    files: list[str],
    labels: list[str],
    *,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    smoothing: float = 0.0,
    target: float | None = None,
):
    assert len(files) == 2
    assert len(labels) == 2

    file_a = files[0]
    file_b = files[1]

    label_a = labels[0]
    label_b = labels[1]

    fs_a, buf_a = wavfile.read(file_a)
    fs_b, buf_b = wavfile.read(file_b)

    assert fs_a == fs_b

    fmax = fmax if fmax != 0.0 else fs_a/2
    nfft = 2**iceil(np.log2(max(buf_a.shape[0], buf_b.shape[0])))
    freqs = np.fft.rfftfreq(nfft, 1/fs_a)

    spectrum_a = np.fft.rfft(buf_a, nfft)
    spectrum_b = np.fft.rfft(buf_b, nfft)

    dB_a = 20*np.log10(np.maximum(np.abs(spectrum_a), 1e-9))
    dB_b = 20*np.log10(np.maximum(np.abs(spectrum_b), 1e-9))

    norm = max(np.max(dB_a), np.max(dB_b))
    dB_a -= norm
    dB_b -= norm

    if smoothing > 0.0:
        dB_a = octave_smoothing(dB_a, fs_a, nfft, smoothing)
        dB_b = octave_smoothing(dB_b, fs_b, nfft, smoothing)

    difference = dB_b-dB_a

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f"{label_a} vs. {label_b}")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    ax0: Axes = ax[0]
    ax0.semilogx(freqs, dB_a, linestyle='-', label=f'{label_a}')
    ax0.semilogx(freqs, dB_b, linestyle='-', label=f'{label_b}')
    ax0.set_title('Spectrum')
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_ylabel('Amplitude [dB]')
    ax0.set_ylim(-60, 0)
    ax0.set_xlim(fmin, fmax)
    ax0.xaxis.set_major_formatter(formatter)
    ax0.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax0.minorticks_on()
    ax0.legend()

    label = f'{label_b}-{label_a}'
    max_diff = np.max(np.abs(difference[(freqs > 10) & (freqs < 20e3)]))
    ax1: Axes = ax[1]
    ax1.semilogx(freqs, difference, linestyle='-', label=label)
    if target and target != 0.0:
        ax1.hlines(target, fmin, fmax, linestyle='--', label=f"Target {target} dB", color='red')
    ax1.set_title('Difference')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude [dB]')
    ax1.set_xlim(fmin, fmax)
    ax1.set_ylim(-max_diff*1.1, max_diff*1.1)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend()


@click.command(name='response', help='Plot frequency response.')
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--fmin', default=1.0)
@click.option('--fmax', default=1000.0)
@click.option('--label_a', default='A')
@click.option('--label_b', default='B')
@click.option('--smoothing', default=0.0)
@click.option('--target', default=0.0)
@click.option('--musical', is_flag=True)
def main(files, fmin, fmax, label_a, label_b, smoothing, target, musical):
    if len(files) == 2:
        plot_response_compare(
            files,
            [label_a, label_b],
            fmin=fmin,
            fmax=fmax,
            smoothing=smoothing,
            target=target,
        )
        plt.show()

    if musical:
        plot_musical_response(files[0], fmin=fmin, fmax=fmax, fraction=12, key_colors=False)
        plt.title('Musical Response')
        plt.show()
