# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.signal import stft

from pffdtd.signals.wavfile import wavread


def plot_spectrogram(x, fs, *, window='hann', min_dB=-100, color_map='gouraud', ax: Axes | None = None):
    if not ax:
        ax = plt.gca()

    nperseg = 128
    nfft = nperseg*32
    frequencies, times, Zxx = stft(x, fs=fs, nperseg=nperseg, nfft=nfft, window=window)

    Zxx_dB = 20*np.log10((np.abs(Zxx)+np.finfo(np.float64).eps)/nfft)
    Zxx_dB -= np.max(Zxx_dB)

    mesh = ax.pcolormesh(times, frequencies, Zxx_dB, shading=color_map, vmin=min_dB, vmax=0)
    ax.figure.colorbar(mesh, label='Amplitude [dB]')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_yscale('log')
    ax.set_ylim(frequencies[1], fs / 2)
    ax.grid(False, which='major')
    ax.grid(False, which='minor')


@click.command(name='spectrogram', help='Spectral decay plot (STFT).')
@click.argument('filename', nargs=1, type=click.Path(exists=True))
@click.option('--color_map', default='gouraud')
@click.option('--min_db', default=-100)
@click.option('--window', default='hann')
def main(filename, color_map, min_db, window):
    fs, ir = wavread(filename)
    ir = ir / np.max(np.abs(ir))

    plt.figure(figsize=(10, 6))
    plot_spectrogram(ir, fs, window=window, min_dB=min_db, color_map=color_map)
    plt.show()
