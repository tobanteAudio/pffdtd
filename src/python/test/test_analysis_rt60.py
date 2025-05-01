# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np

from pffdtd.analysis.rt60 import recommended_rt60


def test_recommended_rt60():
    assert np.allclose(recommended_rt60(45.0), 0.2)
    assert np.allclose(recommended_rt60(50.0), 0.2)
    assert np.allclose(recommended_rt60(100.0), 0.25)
    assert np.allclose(recommended_rt60(200.0), 0.31498)
    assert np.allclose(recommended_rt60(400.0), 0.39685)
    assert np.allclose(recommended_rt60(450.0), 0.4)
    assert np.allclose(recommended_rt60(500.0), 0.4)
