# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import sys

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfilt

from pffdtd.signals.wavfile import wavread
from pffdtd.signals.iir import linkwitz_riley_crossover


def main():
    ir_low_path = sys.argv[1]
    fs_low, ir_low = wavread(ir_low_path)
    print(ir_low.shape)

    ir_high_path = sys.argv[2]
    fs_high, ir_high = wavread(ir_high_path)
    print(ir_high.shape)

    assert fs_low == fs_high
    assert ir_low.shape == ir_high.shape

    sos_low, sos_high = linkwitz_riley_crossover(120, fs_low, order=4)
    filt_low = sosfilt(sos_low, ir_low)
    filt_high = sosfilt(sos_high, ir_high)
    mix = filt_low+filt_high

    # filt_low = ir_low
    # filt_high = ir_high
    # mix = (filt_low+filt_high)/2

    freqs = np.fft.rfftfreq(ir_low.shape[0], 1/fs_low)
    H_low = np.fft.rfft(filt_low)
    H_high = np.fft.rfft(filt_high)
    H_mix = np.fft.rfft(mix)

    _, ax = plt.subplots(1, 1, constrained_layout=True)
    ax: Axes = ax
    ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_low), 1e-6)), linestyle='--', label='Low')
    ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_high), 1e-6)), linestyle='--', label='High')
    ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_mix), 1e-6)), label='Mix')
    ax.set_xlim(10, 1000)
    ax.set_ylim(-70, 10)
    ax.set_ylabel('Amplitude [dB]')
    ax.grid(which='both')
    ax.legend()

    plt.show()


main()
