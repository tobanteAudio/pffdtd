# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np
import pytest

from pffdtd.dsp.pink import generate_pink_noise, pink_noise_slope
from pffdtd.dsp.sine import generate_sine_wave


@pytest.mark.parametrize('fs', [44100, 48000, 96000, 192000])
@pytest.mark.parametrize('duration', [1.0, 5.0, 10.0])
def test_pink_noise(fs, duration):
    rng = np.random.default_rng(123456)
    x = generate_pink_noise(duration, fs, rng=rng)

    # Pink noise should have near-zero mean.
    assert abs(np.mean(x)) < 1e-2

    # Verify PSD falls off ∝ 1/f: slope ≃ -1 on a log-log scale.
    slope = pink_noise_slope(x, fs)
    assert -1.05 < slope < -0.95  # Allow ±5%

    # Check crest factor ≃ 12 dB (±3 dB)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    crest_dB = 20 * np.log10(peak / rms)
    assert 9 < crest_dB < 15


@pytest.mark.parametrize('f', [30.0, 100.0, 440.0])
@pytest.mark.parametrize('fs', [22050, 24000, 44100, 48000, 96000, 192000])
@pytest.mark.parametrize('duration', [1.0, 5.0, 10.0])
def test_sine(f, fs, duration):
    x = generate_sine_wave(f, duration, fs)

    # Should have near-zero mean.
    assert abs(np.mean(x)) < 1e-2

    # Check crest factor ≃ 3 dB (±0.1 dB).
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    crest_dB = 20 * np.log10(peak / rms)
    assert 2.9 < crest_dB < 3.1
