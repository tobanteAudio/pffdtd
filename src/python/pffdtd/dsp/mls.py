# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import max_len_seq

from pffdtd.dsp.wavfile import wavwrite


def generate_max_len_seq(nbits) -> np.ndarray:
    binary: np.ndarray = max_len_seq(nbits)[0]
    return binary.astype(np.float64) * 2 - 1


@click.command(name='mls', help='Generate maximum length sequence.')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--nbits', default=16, type=int)
@click.option('--fs', default=48000, type=int)
@click.option('--plot', is_flag=True)
def main(output, nbits, fs, plot):
    mls_seq = generate_max_len_seq(nbits)
    wavwrite(output, fs, mls_seq)

    if plot:
        plt.plot(np.linspace(0.0, len(mls_seq)/fs, len(mls_seq)), mls_seq)
        plt.grid(which='both')
        plt.show()

        ir = np.fft.rfft(mls_seq)
        freqs = np.fft.rfftfreq(len(mls_seq), d=1/fs)
        plt.semilogx(freqs, 20*np.log10(np.abs(ir)+np.finfo(np.float64).eps))
        plt.grid(which='both')
        plt.show()
