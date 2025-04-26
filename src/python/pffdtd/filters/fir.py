# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np
from scipy.signal import sosfilt, windows


def linear_phase_from_sos(sos, ntaps, window='hann'):
    assert ntaps % 2 != 0

    taps = np.zeros(ntaps)
    taps[ntaps//2] = 1.0

    taps = sosfilt(sos, x=taps)
    taps = sosfilt(sos, x=taps[::-1])[::-1]
    taps *= windows.get_window(window, taps.shape[-1], fftbins=False)

    nfft = max(ntaps, 2048)
    H = np.fft.rfft(taps, nfft)
    H = np.sqrt(np.abs(H))*np.exp(1j*np.angle(H))
    return np.fft.irfft(H, nfft)
