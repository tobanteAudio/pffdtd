# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

from pffdtd.common.wavfile import wavread


@click.command(name='waterfall', help='Waterfall decay plot.')
@click.argument('filename', nargs=1, type=click.Path(exists=True))
@click.option('--color_map', default='gouraud')
@click.option('--min_db', default=-100)
def main(filename, color_map, min_db):
    fs, ir = wavread(filename)
    ir = ir / np.max(np.abs(ir))

    nperseg = 512
    nfft = nperseg*4
    frequencies, times, Zxx = stft(ir, fs=fs, nperseg=nperseg, nfft=nfft)

    Zxx_dB = 20*np.log10((np.abs(Zxx)+np.finfo(np.float64).eps)/nfft)
    Zxx_dB -= np.max(Zxx_dB)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, Zxx_dB, shading=color_map, vmin=min_db, vmax=0)
    plt.colorbar(label='Amplitude [dB]')
    plt.title('Decay Times')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.yscale('log')
    plt.ylim([frequencies[1], fs / 2])
    plt.show()
