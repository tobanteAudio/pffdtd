# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def low_pass(fc, Q, fs):
    omega0 = 2 * np.pi * fc / fs
    d = 1 / Q
    cos0 = np.cos(omega0)
    sin0 = np.sin(omega0)
    beta = 0.5 * ((1 - (d * 0.5) * sin0) / (1 + (d * 0.5) * sin0))
    gamma = (0.5 + beta) * cos0

    b0 = (0.5 + beta - gamma) * 0.5
    b1 = 0.5 + beta - gamma
    b2 = b0

    a0 = 1
    a1 = -2 * gamma
    a2 = 2 * beta

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def butterworth_sos_qs(order, wc=1):
    """
    Calculate the Q factors for each second-order section (SOS) of a Butterworth filter
    of an even order.

    Parameters:
        order (int): The filter order (must be even: 4, 6, 8, ...).
        wc (float): The cutoff frequency (default is 1 for a normalized filter).

    Returns:
        qs (list): A list of Q factors for each SOS.
    """
    if order % 2 != 0:
        raise ValueError('Order must be even.')

    poles = []
    # Calculate all poles using the standard Butterworth formula:
    # s_k = wc * exp(j * (pi/2 + (2k+1)*pi/(2*order))), for k = 0, ..., order-1
    for k in range(order):
        theta = np.pi/2 + (2*k + 1) * np.pi / (2 * order)
        s = wc * np.exp(1j * theta)
        # Only consider poles in the left half-plane
        if s.real < 0:
            poles.append(s)

    # To avoid duplicates from complex conjugate pairs, only keep poles with positive imaginary parts.
    unique_poles = [p for p in poles if p.imag > 0]

    qs = []
    # For each complex conjugate pair, calculate Q
    for s in unique_poles:
        sigma = -s.real  # Make sigma positive since s.real is negative in the LHP
        omega_0 = np.abs(s)  # Natural frequency (should be wc, typically 1)
        Q = omega_0 / (2 * sigma)
        qs.append(Q)

    return qs


def low_pass_sos(x, N, fc, fs):
    qs = butterworth_sos_qs(N)
    for q in qs:
        b, a = low_pass(fc, q, fs)
        x = signal.lfilter(b, a, x)
    return x


def peak_filter(fc, gain, Q, fs):
    assert fs > 0
    assert fc > 0 and fc <= fs * 0.5
    assert Q > 0
    assert gain > 0

    A = np.sqrt(gain)
    omega = (2 * np.pi * max(fc, 2.0)) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A

    b = np.array([1 + alphaTimesA, c2, 1 - alphaTimesA])
    a = np.array([1 + alphaOverA, c2, 1 - alphaOverA])

    b = b / a[0]
    a = a / a[0]

    return signal.tf2sos(b, a)


def group_delay(X, freqs):
    phase = np.unwrap(np.angle(X))
    dphi = np.diff(phase)
    dw = np.diff(freqs)
    return -dphi / dw


def main():
    fs = 96000
    fc = 1000
    order = 2
    freqs = np.fft.rfftfreq(fs, 1/fs)

    impulse = np.zeros(fs)
    impulse[0] = 1.0
    i_fft = np.fft.rfft(impulse)

    l = signal.butter(order, fc, 'lowpass', fs=fs, output='sos')
    l_out = signal.sosfilt(l, impulse.copy())
    l_fft = np.fft.rfft(l_out)

    print(l)

    sos_out = low_pass_sos(impulse.copy(), order, fc=fc, fs=fs)
    sos_fft = np.fft.rfft(sos_out)

    i_mag = 20*np.log10(np.abs(i_fft)+0.0000001)
    l_mag = 20*np.log10(np.abs(l_fft)+0.0000001)
    sos_mag = 20*np.log10(np.abs(sos_fft)+0.0000001)

    sos_gdelay = group_delay(sos_fft, freqs)

    plt.semilogx(freqs, l_mag, label='LFilter')
    plt.semilogx(freqs, sos_mag, label='SOS')
    plt.title(f'{order}th Order Butterworth ({20*np.log10(0.5)*order:.0f}dB)')
    plt.grid(which='both')
    plt.xlim(10, 30_000)
    plt.ylim(-100, 10)
    plt.legend()
    plt.show()

    plt.semilogx(freqs, np.rad2deg(np.angle(i_fft)), label='I Phase')
    plt.semilogx(freqs, np.rad2deg(np.angle(l_fft)), label='L Phase')
    plt.semilogx(freqs, np.rad2deg(np.angle(sos_fft)), label='SOS Phase')
    plt.grid(which='both')
    plt.xlim(10, 30_000)
    # plt.ylim(-100, 10)
    plt.legend()
    plt.show()

    plt.semilogx(freqs[:-1], sos_gdelay, label='SOS Group-Delay')
    plt.grid(which='both')
    # plt.xlim(10, 30_000)
    # plt.ylim(-100, 10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
