# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import numpy as np

from pffdtd.diffusion.diffusor import quadratic_residue_diffusor


def test_quadratic_residue_diffusor():
    w = quadratic_residue_diffusor(5, depth=None)
    assert np.allclose(w, [0, 1, 4, 4, 1])

    w = quadratic_residue_diffusor(7, depth=None)
    assert np.allclose(w, [0, 1, 4, 2, 2, 4, 1])

    w = quadratic_residue_diffusor(11, depth=None)
    assert np.allclose(w, [0, 1, 4, 9, 5, 3, 3, 5, 9, 4, 1])

    w = quadratic_residue_diffusor(5, depth=10)
    assert np.allclose(w, [0, 2.5, 10, 10, 2.5])
