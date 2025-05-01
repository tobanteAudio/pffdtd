# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import numpy as np
import pytest

from pffdtd.absorption.air import sound_attenuation
from pffdtd.absorption.porous import porous_absorber
from pffdtd.dsp.octave import center_frequencies


def test_sound_attenuation():
    assert np.allclose(sound_attenuation(95, 1, 2), 95-6.0205)
    assert np.allclose(sound_attenuation(95, 1, 4), 95-6.0205*2)
    assert np.allclose(sound_attenuation(95, 1, 8), 95-6.0205*3)

    assert np.allclose(sound_attenuation(100, 1, 2), 100-6.0205)
    assert np.allclose(sound_attenuation(100, 1, 4), 100-6.0205*2)
    assert np.allclose(sound_attenuation(100, 1, 8), 100-6.0205*3)


@pytest.mark.parametrize('offset_zeros', [True, False])
@pytest.mark.parametrize(
    'thickness,flow_resistivity,air,expected',
    [
        (0.1, 3000, None, [0.0, 0.00914955, 0.0798299, 0.25952981, 0.53513877, 0.77518233, 0.90934681, 0.95555376, 0.97817786, 0.99485394, 0.99822579]),
        (0.1, 5000, None, [0.0, 0.0, 0.05994547, 0.25596251, 0.59086384, 0.86550292, 0.97620624, 0.97769105, 0.99157503, 0.99930661, 0.999125]),
        (0.1, 8000, None, [0.0, 0.0, 0.05577502, 0.27697129, 0.6626998, 0.93604009, 0.99884517, 0.98875809, 0.99722687, 0.99939574, 0.99926252]),

        (0.1, 3000, 0.05, [0.0, 0.0, 0.0253761, 0.20453878, 0.51929885, 0.80204916, 0.96303949, 0.99265812, 0.94215674, 0.97869725, 0.97120417]),
        (0.1, 5000, 0.05, [0.0, 0.0, 0.04437093, 0.29518879, 0.68539585, 0.93594799, 0.96797934, 0.99916442, 0.97503881, 0.97371309, 0.97207898]),
        (0.1, 8000, 0.05, [0.0, 0.0, 0.09633521, 0.42748118, 0.82984337, 0.99382421, 0.95460156, 0.98919472, 0.98407237, 0.97537423, 0.97443609]),

    ]
)
def test_absorption_porous(offset_zeros, thickness, flow_resistivity, air, expected):
    frequency = center_frequencies(1, 1000, 6, 5)
    absorber = porous_absorber(
        thickness=thickness,
        flow_resistivity=flow_resistivity,
        frequency=frequency,
        air_gap=air,
        angle=45.0,
        offset_zeros=offset_zeros
    )

    assert np.allclose(absorber, expected)
    if offset_zeros:
        assert np.count_nonzero(absorber) == absorber.shape[0]
