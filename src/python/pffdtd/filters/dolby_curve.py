# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


from pffdtd.filters.iir import peak_filter


def dolby_atmos_target_curve_filter(fs):
    # https://gracedesign.com/support/manuals/m908_Atmos_Target_Curve_EQ.pdf
    return np.concatenate([
        peak_filter(16, 10**(-2.3/20), 1.52, fs),
        peak_filter(71, 10**(+1.2/20), 0.44, fs),
        peak_filter(185, 10**(-0.6/20), 0.86, fs),
        peak_filter(5770, 10**(-1.4/20), 0.6, fs),
        peak_filter(20000, 10**(-6.4/20), 0.2, fs),
    ])


def main():
    fs = 48000
    sos = dolby_atmos_target_curve_filter(fs)

    n_ref = fs+1
    sig_ref = np.zeros(n_ref)
    sig_ref[n_ref//2] = 1
    sig_ref = signal.sosfilt(sos, x=sig_ref)
    sig_ref = signal.sosfilt(sos, x=sig_ref[::-1])
    sig_ref *= signal.windows.hann(len(sig_ref))

    freqs_ref = np.fft.rfftfreq(len(sig_ref), 1/fs)
    spectrum_ref = np.fft.rfft(sig_ref)
    spectrum_ref = np.sqrt(np.abs(spectrum_ref))*np.exp(1j*np.angle(spectrum_ref))
    dB_ref = 20*np.log10(np.abs(spectrum_ref)+np.finfo(np.float64).eps)

    n_approx = fs+1
    sig_approx = np.zeros(n_approx)
    sig_approx[n_approx//2] = 1
    sig_approx = signal.sosfilt(sos, x=sig_approx)
    sig_approx = signal.sosfilt(sos, x=sig_approx[::-1])
    sig_approx *= signal.windows.hann(len(sig_approx))

    nfft_approx = max(n_approx, 2048)
    # nfft_approx = n_approx
    freqs_approx = np.fft.rfftfreq(nfft_approx, 1/fs)
    spectrum_approx = np.fft.rfft(sig_approx, nfft_approx)
    spectrum_approx = np.sqrt(np.abs(spectrum_approx))*np.exp(1j*np.angle(spectrum_approx))
    dB_approx = 20*np.log10(np.abs(spectrum_approx)+np.finfo(np.float64).eps)
    impulse_approx = np.fft.irfft(spectrum_approx, nfft_approx)

    resolution = fs/n_approx
    group_delay = (n_approx-1)/2

    plt.semilogx(freqs_ref, dB_ref, label='Ref')
    plt.semilogx(freqs_approx, dB_approx, label=f'Approx ({resolution:.2f} Hz) {1000/fs*group_delay} ms')
    plt.xlim(10, 22_000)
    plt.ylim(-10, 10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title('Dolby Music Curve')
    plt.grid(which='both')
    plt.legend()
    plt.show()

    plt.plot(np.arange(n_approx), impulse_approx[:n_approx])
    # plt.xlim(1.0, 20000.0)
    # plt.ylim(-120.0, 5.0)
    plt.grid(which='both')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
