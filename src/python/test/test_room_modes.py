# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

import pytest

from pffdtd.analysis.room_modes import room_modes


def test_room_modes_2d_excludes_vertical_modes():
    modes = room_modes(1, 1, 1, max_order=2, c=343.2, dimensions=2)

    assert all(mode['p'] == 0 for mode in modes)
    assert [mode['frequency'] for mode in modes[:5]] == pytest.approx([
        171.6,
        171.6,
        242.679047,
        343.2,
        343.2,
    ])


def test_room_modes_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match='two or three dimensions'):
        room_modes(1, 1, 1, dimensions=1)
