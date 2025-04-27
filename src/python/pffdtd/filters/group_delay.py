# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import numpy as np

from pffdtd.filters.iir import minimum_phase_reconstruction


def group_delay_seconds(H, freqs):
    phase = np.unwrap(np.angle(H))
    dphase_df = np.gradient(phase, freqs)
    return -dphase_df / (2 * np.pi)


def excess_group_delay_seconds(H, freqs):
    minphase = minimum_phase_reconstruction(np.abs(H))
    gd_minphase = group_delay_seconds(np.fft.rfft(minphase), freqs)
    return group_delay_seconds(H, freqs)-gd_minphase
