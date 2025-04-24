# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def linkwitz_riley_sos_filter(fc, fs, order=2):
    # Cascading the SOS filters
    # (equivalent to squaring the response of the Butterworth filter)
    lp = signal.butter(order//2, Wn=fc, fs=fs, btype='low', output='sos')
    hp = signal.butter(order//2, Wn=fc, fs=fs, btype='high', output='sos')
    return np.concatenate([lp, lp]), np.concatenate([hp, hp])


def linkwitz_riley_crossover(x: np.ndarray, fs=None, fc=None, order=4):
    lp, hp = linkwitz_riley_sos_filter(fc, fs, order=order)
    return signal.sosfilt(lp, x), signal.sosfilt(hp, x)


def fft_db(x):
    return 20*np.log10(np.abs(np.fft.rfft(x))+np.finfo(np.float64).eps)


def main():
    fc = 800
    order = 4
    fs = 48000*2
    # n = 4096*2
    # impulse = np.zeros(n)
    # impulse[n//2-1] = 1

    # iir = signal.sosfilt(
    #     signal.butter(order, fc, fs=fs, btype='low', output='sos'),
    #     impulse.copy(),
    # )

    # butter = signal.butter(order, fc, fs=fs, btype='low', output='sos')
    # fir = signal.sosfilt(butter, signal.sosfilt(butter, impulse.copy())[::-1])

    # lr_low, lr_high = linkwitz_riley_crossover(impulse.copy(), fs, fc, order)

    # freqs = np.fft.rfftfreq(len(impulse), 1/fs)
    # iir_dB = fft_db(iir)
    # lr_low_dB = fft_db(lr_low)
    # lr_high_dB = fft_db(lr_high)

    # fir_fft = np.fft.rfft(fir)
    # fir_fft = np.sqrt(np.abs(fir_fft))*np.exp(1j*np.angle(fir_fft))
    # fir_dB = 20*np.log10(np.abs(fir_fft)+np.finfo(np.float64).eps)

    # # plt.semilogx(freqs, iir_dB, label='IIR')
    # # plt.semilogx(freqs, fir_dB, label='FIR')
    # plt.semilogx(freqs, lr_low_dB, label='Linkwitz Riley - Low')
    # plt.semilogx(freqs, lr_high_dB, label='Linkwitz Riley - High')

    # plt.xlim(1.0, 30000.0)
    # plt.ylim(-80.0, 5.0)
    # plt.grid(which='both')
    # plt.legend()
    # plt.show()

    fc = 80
    n = fs//16+1
    resolution = fs/n
    group_delay = (n-1)/2
    butter = signal.butter(order, fc, fs=fs, btype='low', output='sos')

    linf = np.zeros(n)
    linf[n//2] = 1.0
    linf = signal.sosfilt(butter, signal.sosfilt(butter, linf.copy())[::-1])
    linf *= signal.windows.hann(n)

    linf_fft = np.fft.rfft(linf)
    linf_fft = np.sqrt(np.abs(linf_fft))*np.exp(1j*np.angle(linf_fft))
    linf_dB = 20*np.log10(np.abs(linf_fft)+np.finfo(np.float64).eps)

    freqs = np.fft.rfftfreq(n, 1/fs)

    plt.semilogx(freqs, linf_dB, label=f'Linear Phase {resolution:.2f} Hz ({1000/fs*group_delay:.2f} ms)')
    plt.xlim(1.0, 20000.0)
    plt.ylim(-120.0, 5.0)
    plt.grid(which='both')
    plt.legend()
    plt.show()

    plt.plot(np.arange(n)/fs, np.fft.irfft(linf_fft, n))
    # plt.xlim(1.0, 20000.0)
    # plt.ylim(-120.0, 5.0)
    plt.grid(which='both')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
