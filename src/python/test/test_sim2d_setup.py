# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import h5py
import numpy as np

from pffdtd.geometry.math import to_ixy
from pffdtd.sim2d.engine import pressure_to_grayscale
from pffdtd.sim2d.setup import generate_source_signal, sim_setup_2d


def test_generate_unit_impulse():
    signal = generate_source_signal(128, 48_000, 2_000, band_limited=False)
    assert signal[0] == 1.0
    assert np.count_nonzero(signal) == 1


def test_generate_band_limited_impulse():
    sample_rate = 48_000
    cutoff = 2_000
    signal = generate_source_signal(4096, sample_rate, cutoff, band_limited=True)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, 1/sample_rate)

    passband = spectrum[np.argmin(np.abs(frequencies-1_000))]
    stopband = spectrum[np.argmin(np.abs(frequencies-4_000))]

    assert np.count_nonzero(signal) > 1
    assert stopband < passband*0.01


def test_band_limited_differentiated_source_is_bipolar(tmp_path):
    def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
        in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
        inx = 2
        iny = 2
        out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
        assert in_mask[inx, iny]
        return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy

    sim_setup_2d(
        sim_dir=tmp_path,
        room=(0.5, 0.5),
        Tc=20,
        rh=50,
        fmax=1_000,
        ppw=10.5,
        duration=0.01,
        refl_coeff=0.99,
        model_factory=model,
        apply_loss=True,
        diff=True,
        band_limited=True,
        image=False,
        verbose=False,
    )

    with h5py.File(tmp_path / 'sim.h5', 'r') as h5f:
        source = h5f['in_sigs'][...]

    assert np.min(source) < 0
    assert np.max(source) > 0


def test_pressure_to_grayscale_uses_symmetric_scale():
    pressure = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    image = pressure_to_grayscale(pressure, scale=1.0)
    assert np.array_equal(image, [[0, 0, 128, 255, 255]])
