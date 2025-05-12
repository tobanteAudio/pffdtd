# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np

from pffdtd.transducer.power import (
    max_sound_pressure,
    power_for_target_spl,
)
from pffdtd.transducer.thiele_small import (
    compliance_equivalent_volume,
    diaphragm_diameter,
    efficiency,
    electrical_q_factor,
    mechanical_compliance,
    mechanical_resistance,
    mechanical_q_factor,
    max_impedance,
    resonance_frequency,
)


def test_max_sound_pressure():
    assert np.allclose(max_sound_pressure(90, 2, 1), 93.01029995)
    assert np.allclose(max_sound_pressure(91, 2, 1), 94.01029995)
    assert np.allclose(power_for_target_spl(93.01029995, 90, 1), 2)
    assert np.allclose(power_for_target_spl(94.01029995, 91, 1), 2)


def test_transducer_thiele_small():
    # Models the workflow when importing a driver in WinISD
    # using the steps from their documentation

    # RSS315HFA-8
    Cms = 0.00027
    Mms = 0.194
    Sd = 0.05067
    BL = 18
    Re = 6.5
    Qms = 2.5

    # Step 1
    Fs = resonance_frequency(Cms, Mms)
    assert np.allclose(Fs, 21.9906)

    # Step 2
    Vas = compliance_equivalent_volume(Sd, Cms)
    Dd = diaphragm_diameter(Sd)
    Qes = electrical_q_factor(Mms, Fs, Re, BL)
    Rms = mechanical_resistance(Sd, Fs, Qms, Vas)
    n0 = efficiency(Fs, Qes, Vas)
    Zmax = max_impedance(Qms, Qes, Re)

    assert np.allclose(Vas, 0.09798)
    assert np.allclose(Dd, 0.253998)
    assert np.allclose(Qes, 0.537758)
    assert np.allclose(Rms, 10.7220)
    assert np.allclose(n0, 0.00189227)
    assert np.allclose(Zmax, 36.718022)

    # Verify
    assert np.allclose(mechanical_compliance(Sd, Vas), Cms)
    assert np.allclose(mechanical_q_factor(Mms, Rms, Fs), Qms)
