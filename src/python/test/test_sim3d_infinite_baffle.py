# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import json

import numpy as np
import pytest

from pffdtd.absorption.admittance import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.common.wavfile import wavread
from pffdtd.geometry.math import point_on_circle
from pffdtd.sim3d.setup import sim_setup_3d
from pffdtd.sim3d.testing import run_engine, skip_if_native_engine_unavailable
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.mark.slow
@pytest.mark.parametrize('engine', ['python', 'cpu'])
def test_sim3d_infinite_baffle(tmp_path, engine):
    skip_if_native_engine_unavailable(engine)

    fmin = 20
    fmax = 1000
    ppw = 10.5
    dx = 343/(fmax*ppw)
    offset = dx*2.0
    root_dir = tmp_path
    sim_dir = root_dir/'cpu'
    model_file = root_dir/'model.json'
    material = 'sabine_01.h5'
    baffle_size = 1.5
    depth = baffle_size/2
    radius = depth-offset*2

    s1 = [baffle_size/2, depth-offset, baffle_size/2]
    r1 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-90))
    r2 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-45))
    r3 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-135))

    r1 = [r1[0], r1[1], s1[2]]
    r2 = [r2[0], r2[1], s1[2]]
    r3 = [r3[0], r3[1], s1[2]]

    model = {
        'mats_hash': {
            'Baffle': {
                'tris': [
                    [0, 2, 1],
                    [0, 3, 2],
                    [1, 5, 4],
                    [1, 2, 5]
                ],
                'pts': [
                    [0.0, depth, 0.0],
                    [baffle_size/2, depth, 0.0],
                    [baffle_size/2, depth, baffle_size],
                    [0.0, depth, baffle_size],
                    [baffle_size, depth, 0.0],
                    [baffle_size, depth, baffle_size]
                ],
                'color': [255, 255, 255],
                'sides': [1, 1, 1, 1]
            }
        },
        'sources': [{'name': 'S1', 'xyz': s1}],
        'receivers': [
            {'name': 'R1', 'xyz': r1},
            {'name': 'R2', 'xyz': r2},
            {'name': 'R3', 'xyz': r3},
        ]
    }

    with open(model_file, 'w') as file:
        json.dump(model, file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.01), root_dir / material)

    sim_setup_3d(
        model_json_file=model_file,
        mat_folder=root_dir,
        mat_files_dict={'Baffle': material},
        diff_source=True,
        duration=3.75,
        fcc_flag=False,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        save_folder=sim_dir,
        save_folder_gpu=sim_dir if engine != 'python' else None,
        bmax=[baffle_size, depth, baffle_size],
        bmin=[0, 0, 0],
        Nprocs=1,
    )

    run_engine(sim_dir=sim_dir, engine=engine)

    process_outputs(
        sim_dir=sim_dir,
        resample_fs=48_000,
        fcut_lowcut=fmin,
        order_lowcut=4,
        fcut_lowpass=fmax,
        order_lowpass=8,
        symmetric_lowpass=True,
        air_abs_filter='none',
        save_wav=True,
        plot_raw=False,
        plot=False,
    )

    fs_1, buf_1 = wavread(sim_dir / 'R001_out_normalised.wav')
    fs_2, buf_2 = wavread(sim_dir / 'R002_out_normalised.wav')
    fs_3, buf_3 = wavread(sim_dir / 'R003_out_normalised.wav')
    assert fs_1 == fs_2
    assert fs_1 == fs_3

    error_2 = 20*np.log10(np.max(np.abs(buf_1-buf_2)))
    error_3 = 20*np.log10(np.max(np.abs(buf_1-buf_3)))
    assert error_2 < -12.0
    assert error_3 < -12.0
