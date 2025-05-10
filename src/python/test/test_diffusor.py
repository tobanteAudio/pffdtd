# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import numpy as np
import pytest

from pffdtd.diffusion.diffusor import (
    diffusor_bandwidth,
    diffusor_dimensions,
    quadratic_residue_diffusor,
)


@pytest.mark.parametrize('c', [343, 343.2, 344])
@pytest.mark.parametrize('fmin', [100, 200, 500])
@pytest.mark.parametrize('fmax', [4000, 5000])
def test_diffusor(c, fmin, fmax):
    dim = diffusor_dimensions(fmin, fmax, c=c)
    bw = diffusor_bandwidth(*dim, c=c)
    assert np.allclose(bw, [fmin, fmax])


def test_quadratic_residue_diffusor():
    w = quadratic_residue_diffusor(5, depth=None)
    assert np.allclose(w, [0, 1, 4, 4, 1])

    w = quadratic_residue_diffusor(7, depth=None)
    assert np.allclose(w, [0, 1, 4, 2, 2, 4, 1])

    w = quadratic_residue_diffusor(11, depth=None)
    assert np.allclose(w, [0, 1, 4, 9, 5, 3, 3, 5, 9, 4, 1])

    w = quadratic_residue_diffusor(5, depth=10)
    assert np.allclose(w, [0, 2.5, 10, 10, 2.5])
