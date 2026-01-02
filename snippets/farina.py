# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from pffdtd.analysis.spectrogram import plot_spectrogram
from pffdtd.signals.convolution import deconvolve
from pffdtd.signals.sine import generate_exponential_sine_sweep


def harmonic_time_shifts(T, f_start, f_end, max_order=5):
    """
    Time offsets for harmonic IRs relative to the *linear* IR.
    From Farina: Δt_n = T * ln(n) / ln(f_end/f_start)
    Returns dict: {order: Δt_seconds}
    order=1 is 0 (linear).
    """
    shifts = {1: 0.0}
    denom = np.log(f_end / f_start)
    for n in range(2, max_order + 1):
        shifts[n] = T * np.log(n) / denom
    return shifts


def main():
    fs = 48_000
    T = 12.0
    fmin = 20
    fmax = 20e3

    shifts = harmonic_time_shifts(T, fmin, fmax, max_order=10)
    for order, delta in shifts.items():
        print(f'{order}: {delta:.6f}')

    sweep, sweep_inv = generate_exponential_sine_sweep(T, fmin, fmax, fs, fade_ms=50)

    ir = convolve(sweep, sweep_inv, mode='full', method='fft')
    ir /= np.max(np.abs(ir)) + 1e-12
    t = np.arange(0, ir.shape[0])/fs-T

    # plot_spectrogram(sweep, fs, window='hann')
    # plt.show()

    # plot_spectrogram(sweep_inv, fs, window='hann')
    # plt.show()

    # plot_spectrogram(ir, fs, window='hann')
    # plt.show()

    plt.plot(t, ir)
    plt.vlines(np.asarray(list(shifts.values())) * -1, -1, 1, colors='red', linestyles='--')
    plt.grid(which='both')
    plt.show()

    plt.plot(t, 20*np.log10(np.maximum(ir, 1e-12)))
    plt.vlines(np.asarray(list(shifts.values())) * -1, -240, 10, colors='red', linestyles='--')
    plt.ylim(-120, 10)
    plt.grid(which='both')
    plt.show()

    H_sweep = np.fft.rfft(sweep)
    H_sweep /= np.max(H_sweep)
    H_sweep = 20*np.log10(np.maximum(np.abs(H_sweep), 1e-12))

    H_sweep_inv = np.fft.rfft(sweep_inv)
    H_sweep_inv /= np.max(H_sweep_inv)
    H_sweep_inv = 20*np.log10(np.maximum(np.abs(H_sweep_inv), 1e-12))

    H_ir = np.fft.rfft(ir)
    H_ir /= np.max(H_ir)
    H_ir = 20*np.log10(np.maximum(np.abs(H_ir), 1e-12))

    h_deconv = deconvolve(sweep, sweep)
    H_deconv = np.fft.rfft(h_deconv)
    H_deconv /= np.max(H_deconv)
    H_deconv = 20*np.log10(np.maximum(np.abs(H_deconv), 1e-12))

    plt.semilogx(np.fft.rfftfreq(sweep.shape[0], 1/fs), H_sweep, label='Sweep')
    plt.semilogx(np.fft.rfftfreq(sweep_inv.shape[0], 1/fs), H_sweep_inv, label='Inverse')
    plt.semilogx(np.fft.rfftfreq(ir.shape[0], 1/fs), H_ir, label='IR')
    plt.semilogx(np.fft.rfftfreq(h_deconv.shape[0], 1/fs), H_deconv, label='Deconv')
    plt.xlim(10, fs/2)
    plt.ylim(-120, 10)
    plt.grid(which='both')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
