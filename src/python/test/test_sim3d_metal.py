# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

import json
import os
import pathlib
import subprocess

import h5py
import numpy as np
import pytest

from pffdtd.absorption.admittance import convert_Sabs_to_Yn, write_freq_ind_mat_from_Yn
from pffdtd.common.metal import metal_device_available
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import sim_setup_3d


def run_native_engine(exe, sim_dir, engine):
    result = subprocess.run(
        args=[str(exe), 'sim3d', '-e', engine, '-p', '32', '-s', str(sim_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    with h5py.File(sim_dir / 'sim_outs.h5', 'r') as f:
        return f['u_out'][...]


def assert_metal_matches_cpu(exe, sim_dir):
    out_cpu = run_native_engine(exe, sim_dir, 'cpu')
    out_metal = run_native_engine(exe, sim_dir, 'metal')

    assert out_metal.shape == out_cpu.shape
    assert np.isfinite(out_metal).all()

    peak = np.max(np.abs(out_cpu))
    assert peak > 0
    assert np.max(np.abs(out_metal - out_cpu)) < peak*1e-5


@pytest.mark.slow
@pytest.mark.skipif(not metal_device_available(), reason='No Metal device is available')
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 3D engine not available')
def test_sim3d_metal_engine(tmp_path):
    sim_dir = tmp_path / 'sim'
    model_file = tmp_path / 'model.json'
    material_file = tmp_path / 'material.h5'

    room = RoomModelBuilder(length=0.7, width=0.6, height=0.8)
    room.add_source('S1', [0.2, 0.2, 0.2])
    room.add_receiver('R1', [0.4, 0.45, 0.5])
    room.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.2), material_file)

    sim_setup_3d(
        model_json_file=model_file,
        mat_folder=tmp_path,
        mat_files_dict={
            'Ceiling': material_file.name,
            'Floor': material_file.name,
            'Walls': material_file.name,
        },
        diff_source=True,
        duration=0.005,
        fcc_flag=False,
        fmax=500.0,
        PPW=10.5,
        insig_type='impulse',
        save_folder=sim_dir,
        save_folder_gpu=sim_dir,
        Nprocs=1,
    )

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.is_file()

    assert_metal_matches_cpu(exe, sim_dir)


@pytest.mark.slow
@pytest.mark.skipif(not metal_device_available(), reason='No Metal device is available')
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 3D engine not available')
def test_sim3d_metal_engine_with_rigid_boundaries(tmp_path):
    sim_dir = tmp_path / 'sim'
    model_file = tmp_path / 'model.json'

    model = {
        'mats_hash': {
            '_RIGID': {
                'tris': [[0, 2, 1], [0, 3, 2], [1, 5, 4], [1, 2, 5]],
                'pts': [
                    [0.0, 0.35, 0.0],
                    [0.3, 0.35, 0.0],
                    [0.3, 0.35, 0.8],
                    [0.0, 0.35, 0.8],
                    [0.6, 0.35, 0.0],
                    [0.6, 0.35, 0.8],
                ],
                'color': [255, 255, 255],
                'sides': [0, 0, 0, 0],
            },
        },
        'sources': [{'name': 'S1', 'xyz': [0.3, 0.2, 0.4]}],
        'receivers': [{'name': 'R1', 'xyz': [0.3, 0.5, 0.4]}],
    }
    with open(model_file, 'w') as f:
        json.dump(model, f)

    sim_setup_3d(
        model_json_file=model_file,
        mat_folder=tmp_path,
        mat_files_dict={},
        diff_source=True,
        duration=0.005,
        fcc_flag=False,
        fmax=500.0,
        PPW=10.5,
        insig_type='impulse',
        save_folder=sim_dir,
        save_folder_gpu=sim_dir,
        bmin=[0, 0, 0],
        bmax=[0.6, 0.7, 0.8],
        Nprocs=1,
    )

    with h5py.File(sim_dir / 'vox_out.h5', 'r') as f:
        assert np.all(f['mat_bn'][...] == -1)

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.is_file()

    assert_metal_matches_cpu(exe, sim_dir)
