# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import os
import pathlib
import subprocess

import h5py
import numpy as np
import pytest

from pffdtd.geometry.math import to_ixy
from pffdtd.common.metal import metal_device_available
from pffdtd.sim2d.setup import sim_setup_2d


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 2D engine not available')
@pytest.mark.skipif(not metal_device_available(), reason='No Metal device is available')
def test_sim2d_metal(tmp_path):
    def multi_source_model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
        in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
        in_ixy = [to_ixy(2, 2, Nx, Ny), to_ixy(2, 3, Nx, Ny)]
        out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
        assert in_mask.flat[in_ixy].all()
        return in_mask, in_ixy, out_ixy

    sim_setup_2d(
        sim_dir=tmp_path,
        room=(0.5, 0.65),
        Tc=20,
        rh=50,
        fmax=1000.0,
        ppw=10.5,
        duration=0.25,
        refl_coeff=0.991,
        model_factory=multi_source_model,
        apply_loss=True,
        diff=True,
        image=False,
        verbose=False,
    )

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.is_file()

    for engine in ('cpu', 'metal'):
        args = [str(exe), 'sim2d', '-e', engine, '-p', '32', '-s', str(tmp_path), '-o', f'out-{engine}.h5']
        result = subprocess.run(args=args, capture_output=True, text=True, check=True)
        assert result.returncode == 0

    with h5py.File(tmp_path / 'out-cpu.h5', 'r') as f:
        out_cpu = f['out'][...]
    with h5py.File(tmp_path / 'out-metal.h5', 'r') as f:
        out_metal = f['out'][...]

    assert out_metal.shape == out_cpu.shape
    np.testing.assert_allclose(out_cpu, out_metal, rtol=0, atol=10**(-96/20), equal_nan=False)
