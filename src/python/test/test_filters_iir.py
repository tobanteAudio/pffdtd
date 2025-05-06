# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np
import pytest
from scipy.signal import butter, sosfilt, unit_impulse

from pffdtd.signals.iir import butterworth_Qs, low_pass, peak_filter


def test_butterworth_Qs():
    assert np.allclose(butterworth_Qs(2), [1/np.sqrt(2)])
    assert np.allclose(butterworth_Qs(4), [1.30656296, 0.5411961])
    assert np.allclose(butterworth_Qs(6), [1.93185165, 0.70710678, 0.51763809])
    assert np.allclose(butterworth_Qs(8), [2.56291544, 0.89997622, 0.60134488, 0.50979557])


@pytest.mark.parametrize('fc', [30, 100, 440, 2000])
@pytest.mark.parametrize('fs', [24000, 44100, 48000, 88200, 96000])
def test_low_pass(fc, fs):
    actual = low_pass(fc, butterworth_Qs(2)[0], fs)
    expected = butter(2, fc, btype='low', fs=fs, output='sos')
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('fc', [30, 100, 440, 2000])
@pytest.mark.parametrize('fs', [24000, 44100, 48000, 88200, 96000])
@pytest.mark.parametrize('gain_dB', [1.0, 2.0, 6.0, 12.0])
def test_peak_filter(fc, fs, gain_dB):
    nfft = fs*2
    sos = peak_filter(fc, 10**(gain_dB/20), 1/np.sqrt(2), fs)
    y: np.ndarray = sosfilt(sos, unit_impulse(nfft))
    freqs: np.ndarray = np.fft.rfftfreq(nfft, 1/fs)
    H = np.fft.rfft(y, nfft)
    H_dB = 20*np.log10(np.maximum(np.abs(H), 1e-12))
    bin = H_dB[np.abs(freqs-fc).argmin()]
    assert np.allclose(bin, gain_dB)
