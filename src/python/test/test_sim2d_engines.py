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
from pffdtd.sim2d.setup import sim_setup_2d


def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
    inx = 2
    iny = 2
    out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
    assert in_mask[inx, iny]
    return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 2D engine not available')
def test_sim2d_engines(tmp_path):
    sim_setup_2d(
        sim_dir=tmp_path,
        room=(2, 2),
        Tc=20,
        rh=50,
        fmax=1000.0,
        ppw=10.5,
        duration=6.0,
        refl_coeff=0.991,
        model_factory=model,
        apply_loss=True,
        diff=True,
        image=True,
        verbose=True,
    )

    runner = CliRunner()
    args = ['sim2d', 'run', '--sim_dir', str(tmp_path), '--out', 'out-py.h5']
    result = runner.invoke(cli, args)
    assert result.exit_code == 0

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.exists()
    assert exe.is_file()

    result = subprocess.run(
        args=[str(exe), 'sim2d', '-p', '64', '-s', str(tmp_path), '-o', 'out-cpp.h5'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    out_file = h5py.File(tmp_path / 'out-py.h5', 'r')
    out_py = out_file['out'][...]
    out_file.close()

    out_file = h5py.File(tmp_path / 'out-cpp.h5', 'r')
    out_cpp = out_file['out'][...]
    out_file.close()

    assert not np.isnan(out_py).any()
    assert not np.isinf(out_py).any()
    assert not np.isnan(out_cpp).any()
    assert not np.isinf(out_cpp).any()
    assert np.allclose(out_py, out_cpp)
