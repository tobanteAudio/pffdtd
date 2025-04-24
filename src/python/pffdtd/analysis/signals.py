# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import max_len_seq

from pffdtd.common.wavfile import wavwrite


def generate_max_len_seq(nbits) -> np.ndarray:
    binary: np.ndarray = max_len_seq(nbits)[0]
    return binary.astype(np.float64) * 2 - 1


@click.command(name='signals', help='Generate test signals')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--nbits', default=16, type=int)
@click.option('--fs', default=48000, type=int)
@click.option('--plot', is_flag=True)
def main(output, nbits, fs, plot):
    mls = generate_max_len_seq(nbits)
    wavwrite(output, fs, mls)

    if plot:
        plt.plot(np.linspace(0.0, len(mls)/fs, len(mls)), mls)
        plt.grid(which='both')
        plt.show()

        ir = np.fft.rfft(mls)
        freqs = np.fft.rfftfreq(len(mls), d=1/fs)
        plt.semilogx(freqs, 20*np.log10(np.abs(ir)+0.0000001))
        plt.grid(which='both')
        plt.show()
