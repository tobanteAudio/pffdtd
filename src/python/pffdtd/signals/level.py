# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np


def crest_factor(x):
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    return peak/rms


def normalize_to_RMS_dBFS(
    x: np.ndarray,
    target_dBFS: float,
    *,
    x_ref: np.ndarray | None = None,
) -> np.ndarray:
    if x_ref is None:
        rms = np.sqrt(np.mean(x**2))
    else:
        rms = np.sqrt(np.mean(x_ref**2))

    current_dB = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dB
    gain = 10**(gain_dB / 20)
    return x * gain
