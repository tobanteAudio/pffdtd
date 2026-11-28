# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import os
import pathlib
import subprocess

import h5py
import numpy as np
import pytest

from pffdtd.absorption.admittance import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.sim3d.engine import EnginePython3D
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import sim_setup_3d
from pffdtd.sim3d.testing import run_engine
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.fixture(scope='module')
def exe_path():
    env = os.environ.get('PFFDTD_ENGINE_EXE')
    if not env:
        pytest.skip('Native 3D engine not available')
    exe = pathlib.Path(env).absolute()
    assert exe.is_file()
    return exe


@pytest.fixture(
    scope='module',
    params=[
        pytest.param({}, id='baseline'),
        pytest.param({'diff': True}, id='diff'),
        pytest.param({'ppw': 7.75}, id='ppw'),
        pytest.param({'fcc': True, 'ppw': 7.75}, id='fcc'),
        pytest.param({'dim': (1.0, 1.0, 1.0)}, id='cube-room'),
        pytest.param({'fmax': 2000.0, 'duration': 0.05}, id='high-frequency'),
    ],
)
def sim_config(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp('python_vs_engine_3d')
    config = {
        'dim': (1.0, 2.0, 1.5),
        'fmax': 800,
        'ppw': 10.5,
        'fcc': False,
        'diff': True,
        'duration': 0.15,
        **request.param,
    }

    root_dir = tmp_path
    model_file = root_dir/'model.json'
    material = 'material.h5'

    L = config['dim'][0]
    W = config['dim'][1]
    H = config['dim'][2]
    room = RoomModelBuilder(L, W, H)
    room.add_source('S1', [0.25, 0.25, 0.25])
    room.add_receiver('R1', [W-0.25, L-0.25, H-0.25])
    room.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.03), root_dir / material)

    return {
        'model_json_file': model_file,
        'mat_folder': root_dir,
        'mat_files_dict': {
            'Ceiling': material,
            'Floor': material,
            'Walls': material,
        },
        'diff_source': config['diff'],
        'duration': config['duration'],
        'fcc_flag': config['fcc'],
        'fmax': config['fmax'],
        'PPW': config['ppw'],
        'insig_type': 'impulse',
        'save_folder': tmp_path,
        'save_folder_gpu': None,
        'Nprocs': 1,
    }


@pytest.fixture(scope='module')
def python_output(sim_config):
    sim_dir = sim_config['save_folder']
    sim_setup_3d(**sim_config)
    eng = EnginePython3D(sim_dir)
    eng.run_all(1)
    eng.save_outputs()

    with h5py.File(sim_dir / 'sim_outs.h5', 'r') as f:
        out_py = np.array(f['u_out'][...], copy=True)
        out_py.setflags(write=False)

    return sim_dir, out_py


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
def test_sim3d_python_vs_engine(engine, exe_path, sim_config, python_output):
    if engine in ['metal', 'sycl'] and sim_config['fcc_flag']:
        pytest.skip(f"'{engine}' engine does not support the fcc_flag")
    sim_dir, out_py = python_output

    args = [str(exe_path), 'sim3d', '-e', engine, '-p', '64', '-s', str(sim_dir)]
    result = subprocess.run(args=args, capture_output=True, check=True)
    assert result.returncode == 0

    with h5py.File(sim_dir / f'sim_outs.h5', 'r') as f:
        out = f['u_out'][...]
    assert out.shape == out_py.shape
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, out_py, rtol=0, atol=10**(-144/20))
