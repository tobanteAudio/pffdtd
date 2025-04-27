# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np
from scipy import signal


def butterworth_Qs(order, wc=1):
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


def peak_filter(fc, gain, Q, fs):
    assert fs > 0
    assert fc > 0
    assert fc <= fs * 0.5
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


def minimum_phase_reconstruction(M_half: np.ndarray) -> np.ndarray:
    """
    Reconstructs a minimum phase impulse response with a magnitude
    response matching M

    Parameters:
        - M_half: Of length (N//2 + 1), containing magnitudes at frequencies 0, 2π/N, 4π/N, …, π (Nyquist).

    Returns:
        Minimum phase impulse response
    """
    # infer full FFT length N (must be even)
    N = (len(M_half) - 1) * 2

    # 1. Reconstruct full, even‐symmetric magnitude spectrum
    #   bins 0 … N/2
    #   then bins N/2−1 … 1
    M_full = np.concatenate([
        M_half,
        M_half[-2:0:-1]   # skip the last (Nyquist) and the first (DC)
    ])

    # 2. Log‐magnitude and real cepstrum
    eps = 1e-12                                 # avoid log(0)
    L = np.log(np.maximum(M_full, eps))         # length-N real, symmetric
    c = np.fft.ifft(L).real                     # real cepstrum, length N

    # 3. Build the minimum‐phase cepstrum
    c_min = np.zeros_like(c)
    c_min[0] = c[0]                             # keep the DC term
    # double the causal part 1 … N/2−1
    c_min[1:N//2] = 2 * c[1:N//2]
    # if you want to preserve the Nyquist term (for even N), uncomment:
    c_min[N//2] = c[N//2]

    # 4. Re‐synthesize the complex spectrum
    #    H_min[k] = exp( FFT{c_min} )
    H_min = np.exp(np.fft.fft(c_min))
    h = np.fft.ifft(H_min)
    return h.real
