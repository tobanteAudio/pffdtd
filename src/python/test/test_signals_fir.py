# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import pytest

import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import butter,  freqz, sosfreqz

from pffdtd.signals.fir import linear_phase_from_sos


@pytest.mark.parametrize('ntaps', [511, 1023, 4095])
def test_linear_phase_from_sos(ntaps):
    fs = 1.0
    sos = butter(6, 0.2, btype='low', output='sos', fs=fs)
    h = linear_phase_from_sos(sos, ntaps=ntaps)

    # Length
    assert len(h) == ntaps

    # Symmetry
    assert_allclose(h, h[::-1], rtol=1e-4, atol=1e-6)

    # Magnitude response
    _, H_iir = sosfreqz(sos, worN=8192, fs=fs)
    _, H_fir = freqz(h, worN=8192, fs=fs)
    mag_iir = np.abs(H_iir)
    mag_fir = np.abs(H_fir)
    assert_allclose(mag_fir, mag_iir, rtol=5e-3, atol=5e-4)
