# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

import numpy as np
import pytest

from pffdtd.absorption.air import (
    Air,
    air_absorption,
    air_density,
    apply_modal_filter,
    apply_ola_filter,
    apply_visco_filter,
    sound_velocity,
    wave_number_in_air,
)


def test_air_properties():
    assert air_density(1.0, 0.0) == pytest.approx(1.29228367)
    assert sound_velocity(0.0) == pytest.approx(331.461469)

    air = Air(
        temperature=np.float64(20),
        pressure=np.float64(1),
        density=np.float64(1.2),
        velocity=np.float64(343),
        impedance=np.float64(411.6),
        tau_over_c=np.float64(2*np.pi/343),
    )
    assert wave_number_in_air(air, 1000) == pytest.approx(2*np.pi*1000/343)


def test_air_absorption_standard_conditions():
    frequencies = np.array([125.0, 1000.0, 8000.0])
    result = air_absorption(frequencies, 20.0, 50.0)

    assert result.c == pytest.approx(343.2)
    assert result.frO == pytest.approx(35413.8596168)
    assert result.frN == pytest.approx(331.8504944)
    assert np.allclose(
        result.absfull_dB,
        [0.00043990, 0.00466625, 0.10534367],
        rtol=1e-6,
    )
    assert np.allclose(result.absfull_Np, result.absfull_dB*np.log(10)/20)
    assert np.allclose(
        result.absfull_dB,
        result.absClRo_dB+result.absVibO_dB+result.absVibN_dB,
    )


def test_air_absorption_uses_ambient_pressure():
    frequencies = np.array([125.0, 1000.0, 8000.0])
    low_pressure = air_absorption(frequencies, 20.0, 50.0, 80.0)
    standard_pressure = air_absorption(frequencies, 20.0, 50.0, 101.325)

    assert low_pressure.frO != pytest.approx(standard_pressure.frO)
    assert low_pressure.frN != pytest.approx(standard_pressure.frN)
    assert not np.allclose(
        low_pressure.absfull_dB, standard_pressure.absfull_dB
    )


@pytest.mark.parametrize('pressure', [0.0, -1.0, 201.0])
def test_air_absorption_rejects_invalid_pressure(pressure):
    with pytest.raises(AssertionError):
        air_absorption([1000.0], 20.0, 50.0, pressure)


def test_air_absorption_rejects_negative_frequency():
    with pytest.raises(AssertionError):
        air_absorption([-1.0], 20.0, 50.0)


def test_modal_filter_preserves_zero_signal():
    signal = np.zeros(16)
    filtered = apply_modal_filter(signal, 8000, 20.0, 50.0)
    assert np.array_equal(filtered, signal)


def test_ola_filter_preserves_zero_signal():
    filtered = apply_ola_filter(np.zeros(16), 8000, 20.0, 50.0, Nw=16)
    assert filtered.ndim == 1
    assert np.count_nonzero(filtered) == 0


def test_visco_filter_handles_signal_shorter_than_unfiltered_prefix():
    signal = np.arange(8, dtype=float)
    filtered = apply_visco_filter(signal, 8000, 20.0, 50.0)

    assert filtered.shape[0] >= signal.shape[0]
    assert np.array_equal(filtered[:signal.size], signal)
    assert np.count_nonzero(filtered[signal.size:]) == 0
