# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np
import pytest
from scipy.signal import welch

from pffdtd.signals.pink import generate_pink_noise


@pytest.mark.parametrize('fs', [44100, 48000, 96000, 192000])
@pytest.mark.parametrize('duration', [1.0, 5.0, 10.0])
def test_mean_close_to_zero(fs, duration):
    """Pink noise should have near-zero mean."""
    x = generate_pink_noise(duration, fs)
    assert abs(np.mean(x)) < 1e-2


@pytest.mark.parametrize('fs', [44100, 48000])
@pytest.mark.parametrize('duration', [1.0, 5.0, 10.0])
def test_psd_slope(fs, duration):
    """
    Verify PSD falls off ∝ 1/f:
    slope ≃ -1 on a log-log scale.
    """
    x = generate_pink_noise(duration, fs)

    # Estimate PSD
    f, Pxx = welch(x, fs, nperseg=4096)

    # Restrict to 20 Hz–20 kHz
    mask = (f >= 20) & (f <= 20000)
    logf = np.log10(f[mask])
    logP = np.log10(Pxx[mask])
    slope, _ = np.polyfit(logf, logP, 1)

    # Ideal slope = –1; allow ±5%
    assert -1.05 < slope < -0.95


@pytest.mark.parametrize('fs', [44100, 48000, 96000, 192000])
@pytest.mark.parametrize('duration', [1.0, 5.0, 10.0])
def test_crest_factor(fs, duration):
    """Check crest factor ≃ 12 dB (±3 dB)."""
    x = generate_pink_noise(duration, fs)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    crest_dB = 20 * np.log10(peak / rms)
    assert 9 < crest_dB < 15
