# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import os
import pathlib
import subprocess

from click.testing import CliRunner
import h5py
import numpy as np
import pytest

from pffdtd.cli import main as cli
from pffdtd.geometry.math import to_ixy
from pffdtd.sim2d.setup import sim_setup_2d


@pytest.fixture(scope='session')
def exe_path():
    env = os.environ.get('PFFDTD_ENGINE_EXE')
    if not env:
        pytest.skip('Native 2D engine not available')
    exe = pathlib.Path(env).absolute()
    assert exe.is_file()
    return exe


@pytest.fixture(
    scope='session',
    params=[
        pytest.param({}, id='baseline'),
        pytest.param({'fmax': 10000.0, 'duration': 0.05}, id='high-frequency'),
        pytest.param({'room': (4.0, 4.0)}, id='large-room'),
        pytest.param({'room': (2.0, 4.0)}, id='rectangular-room'),
        pytest.param({'apply_loss': False}, id='lossless'),
        pytest.param({'refl_coeff': 0.8}, id='high-absorption'),
        pytest.param({'diff': True}, id='diff'),
    ],
)
def sim_config(request):
    def _model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
        in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
        inx = 2
        iny = 2
        out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
        assert in_mask[inx, iny]
        return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy

    room_size = 2.0
    return {
        'room': (room_size, room_size),
        'Tc': 20,
        'rh': 50,
        'fmax': 1000.0,
        'ppw': 10.5,
        'duration': 0.25,
        'refl_coeff': 0.991,
        'model_factory': _model,
        'apply_loss': True,
        'diff': False,
        'image': False,
        'verbose': False,
        **request.param,
    }


@pytest.fixture(scope='session')
def python_output(tmp_path_factory, sim_config):
    tmp_path = tmp_path_factory.mktemp('python_vs_engine')
    sim_setup_2d(sim_dir=tmp_path, **sim_config)

    runner = CliRunner()
    args = ['sim2d', 'run', '--sim_dir', str(tmp_path), '--out', 'out-py.h5']
    result = runner.invoke(cli, args)
    assert result.exit_code == 0

    with h5py.File(tmp_path / 'out-py.h5', 'r') as f:
        out_py = np.array(f['out'][...], copy=True)
        out_py.setflags(write=False)
    return out_py, tmp_path


def engines_to_test() -> list[str]:
    engines = ['cpu']
    extra = os.environ.get('PFFDTD_EXTRA_TEST_ENGINES')
    if extra:
        for engine in extra.split(';'):
            # Metal is excluded, because it doesn't support float64
            assert engine in ['cuda', 'sycl']
            engines.append(engine)
    return engines


@pytest.mark.slow
@pytest.mark.parametrize('engine', engines_to_test())
def test_sim2d_python_vs_engine(engine, exe_path, sim_config, python_output):
    out_py, sim_path = python_output
    sim_setup_2d(sim_dir=sim_path, **sim_config)

    args = [str(exe_path), 'sim2d', '-e', engine, '-p', '64', '-s', str(sim_path), '-o', f'out-{engine}.h5']
    result = subprocess.run(args=args, capture_output=True, text=True, check=True)
    assert result.returncode == 0

    with h5py.File(sim_path / f'out-{engine}.h5', 'r') as f:
        out = f['out'][...]
    assert out.shape == out_py.shape
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, out_py, rtol=0, atol=10**(-144/20))
