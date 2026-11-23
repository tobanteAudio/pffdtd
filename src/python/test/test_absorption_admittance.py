# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

import h5py
import numpy as np
import pytest

from pffdtd.absorption.admittance import (
    _from_DEF,
    _to_DEF,
    compute_Rf_from_DEF,
    convert_R_to_Yn,
    convert_R_to_Zn,
    convert_Sabs_to_Yn,
    convert_Yn_to_R,
    convert_nabs_to_R,
    fit_to_Sabs_oct_11,
    read_mat_DEF,
    write_freq_dep_mat,
    write_freq_ind_mat_from_Yn,
    write_freq_ind_mat_from_Zn,
)


def test_absorption_and_admittance_conversions():
    absorption = np.array([0.0, 0.36, 1.0])
    assert np.allclose(convert_nabs_to_R(absorption), [1.0, 0.8, 0.0])

    admittance = np.array([0.25, 1.0, 4.0])
    reflection = convert_Yn_to_R(admittance)
    assert np.allclose(reflection, [0.6, 0.0, -0.6])
    assert np.allclose(convert_R_to_Yn(reflection), admittance)
    assert np.allclose(convert_R_to_Zn(reflection), 1/admittance)


@pytest.mark.parametrize('absorption', [-0.01, 1.01])
def test_normal_absorption_rejects_values_outside_unit_interval(absorption):
    with pytest.raises(AssertionError):
        convert_nabs_to_R(absorption)


@pytest.mark.parametrize('absorption', [0.0, 0.1, 0.5, 0.9, 0.9512])
def test_sabine_absorption_conversion_inverts_paris_formula(absorption):
    admittance = convert_Sabs_to_Yn(absorption)

    if absorption == 0:
        assert admittance == 0
        return

    reconstructed = 8*admittance*(
        1+admittance/(1+admittance)
        - 2*admittance*np.log((admittance+1)/admittance)
    )
    assert reconstructed == pytest.approx(absorption, abs=1e-6)


def test_sabine_absorption_rejects_negative_value():
    with pytest.raises(AssertionError):
        convert_Sabs_to_Yn(-0.01)


def test_def_parameter_conversion_round_trip():
    admittance = np.array([0.1, 0.25])
    bandwidth = np.array([10.0, 20.0])
    resonance = np.array([100.0, 400.0])

    damping, resistance, stiffness = _to_DEF(
        admittance, bandwidth, resonance
    )
    actual = _from_DEF(damping, resistance, stiffness)

    for result, expected in zip(actual, (admittance, bandwidth, resonance)):
        assert np.allclose(result, expected)


def test_compute_reflection_from_def():
    jw = 1j*np.array([1.0, 2.0])
    damping = np.array([0.0, 1.0])
    resistance = np.array([2.0, 3.0])
    stiffness = np.array([0.0, 4.0])

    reflection, admittance, impedance, branch_reflection = compute_Rf_from_DEF(
        jw, damping, resistance, stiffness
    )

    expected_impedance = (
        jw[:, None]*damping + resistance + stiffness/jw[:, None]
    )
    expected_admittance = np.sum(1/expected_impedance, axis=1)
    assert np.allclose(impedance, expected_impedance)
    assert np.allclose(admittance, expected_admittance)
    assert np.allclose(
        reflection, (1-expected_admittance)/(1+expected_admittance)
    )
    assert np.allclose(
        branch_reflection, (expected_impedance-1)/(expected_impedance+1)
    )


@pytest.mark.parametrize(
    'writer,value,expected',
    [
        (write_freq_ind_mat_from_Zn, 4.0, [[0.0, 4.0, 0.0]]),
        (write_freq_ind_mat_from_Yn, 0.25, [[0.0, 4.0, 0.0]]),
    ],
)
def test_write_frequency_independent_material(tmp_path, writer, value, expected):
    filename = tmp_path/'material.h5'
    writer(value, filename)
    assert np.array_equal(read_mat_DEF(filename), expected)


def test_write_frequency_dependent_material(tmp_path):
    filename = tmp_path/'material.h5'
    expected = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0]])

    write_freq_dep_mat(expected, filename)

    assert np.array_equal(read_mat_DEF(filename), expected)
    with h5py.File(filename, 'r') as material:
        assert list(material) == ['DEF']


def test_fit_sabine_octave_bands_writes_valid_def(tmp_path):
    filename = tmp_path/'fitted.h5'

    fit_to_Sabs_oct_11(np.full(11, 0.2), filename)

    result = read_mat_DEF(filename)
    assert result.shape == (11, 3)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)
