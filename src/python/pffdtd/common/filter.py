# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

import numpy as np
from scipy.signal import butter, bilinear_zpk, zpk2sos, sosfilt, lfilter


def apply_lowcut(y, fs, fcut, order, apply_int) -> np.ndarray:
    dt = 1/fs

    if fcut > 0:
        if apply_int:
            # design combined butter filter with integrator
            Wn = fcut*2*np.pi
            z, p, k = butter(order, Wn, btype='high',
                             analog=True, output='zpk')
            assert np.all(z == 0.0)
            z = z[1:]  # remove one zero
            zd, pd, kd = bilinear_zpk(z, p, k, 1/dt)
            sos = zpk2sos(zd, pd, kd)
        else:
            # design digital high-pass
            sos = butter(order, 2*dt*fcut, btype='high', output='sos')
        return sosfilt(sos, y)

    if apply_int:
        # shouldn't really use this without lowcut, but here in case
        b = dt/2*np.array([1, 1])
        a = np.array([1, -1])
        return lfilter(b, a, y)

    return np.copy(y)


def apply_lowpass(y: np.ndarray, fs: float, fcut: float, order: int = 8, symmetric=True) -> np.ndarray:
    y_out = np.copy(y)

    if symmetric:  # will be run twice
        assert order % 2 == 0
        order = int(order//2)

    # design digital high-pass
    sos = butter(order, fcut, btype='low', output='sos', fs=fs)
    y_out = sosfilt(sos, y_out)
    if symmetric:  # runs again, time reversed
        y_out = sosfilt(sos, y_out[:, ::-1])[:, ::-1]

    return y_out
