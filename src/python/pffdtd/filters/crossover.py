# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

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
