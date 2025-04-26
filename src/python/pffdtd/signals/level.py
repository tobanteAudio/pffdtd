# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np


def crest_factor(x):
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    return peak/rms


def normalize_to_RMS_dBFS(x, target_dBFS):
    rms = np.sqrt(np.mean(x**2))
    current_dB = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dB
    gain = 10**(gain_dB / 20)
    return x * gain
