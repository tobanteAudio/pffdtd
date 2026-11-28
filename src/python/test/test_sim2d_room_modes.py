# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import os
import pathlib
import subprocess

from click.testing import CliRunner
import h5py
import numpy as np
import pytest

from pffdtd.geometry.math import to_ixy
from pffdtd.cli import main as cli
from pffdtd.common.metal import metal_device_available
from pffdtd.signals.music import cents_deviation
from pffdtd.sim2d.engine import pressure_to_grayscale
from pffdtd.sim2d.setup import generate_source_signal, sim_setup_2d


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 2D engine not available')
@pytest.mark.parametrize('room_size', [1.0, 1.25])
def test_sim2d_room_modes(tmp_path, room_size):
    def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
        in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
        inx = 2
        iny = 2
        out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
        assert in_mask[inx, iny]
        return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy

    sim_setup_2d(
        sim_dir=tmp_path,
        room=(room_size, room_size),
        Tc=20,
        rh=50,
        fmax=2000.0,
        ppw=10.5,
        duration=2.0,
        refl_coeff=0.991,
        model_factory=model,
        apply_loss=True,
        diff=False,
        image=True,
        verbose=True,
    )

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.is_file()

    args = [str(exe), 'sim2d', '-p', '64', '-s', str(tmp_path), '-o', 'out.h5']
    result = subprocess.run(args=args, capture_output=True, text=True, check=True)
    assert result.returncode == 0

    with h5py.File(tmp_path / 'constants.h5', 'r') as f:
        c = float(f['c'][...])
        fs = float(f['fs'][...])
    with h5py.File(tmp_path / 'out.h5', 'r') as f:
        out = f['out'][...]

    assert np.isfinite(out).all()

    expected_mode = c/(2*room_size)
    spectrum = np.abs(np.fft.rfft(out[0]*np.hanning(out.shape[-1])))
    frequencies = np.fft.rfftfreq(out.shape[-1], 1/fs)
    mode_band = (frequencies > 0.75*expected_mode) & (frequencies < 1.35*expected_mode)
    measured_mode = frequencies[mode_band][np.argmax(spectrum[mode_band])]
    assert abs(cents_deviation(expected_mode, measured_mode)) < 10.0
