# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import numpy as np
import pytest

from pffdtd.absorption.admittance import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.analysis.room_modes import detect_room_modes, find_nearest
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import sim_setup_3d
from pffdtd.sim3d.testing import run_engine, skip_if_native_engine_unavailable
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.mark.slow
@pytest.mark.parametrize('engine', ['python', 'cpu'])
@pytest.mark.parametrize(
    'room,fmax,ppw,fcc,dx_scale,tolerance_pct',
    [
        ((2.8, 2.076, 1.48), 400, 10.5, False, 2, 1.7),
        ((3.0, 1.0, 2.0), 800, 7.75, True, 3, 3.8),
    ]
)
def test_sim3d_detect_room_modes(tmp_path, engine, room, fmax, ppw, fcc, dx_scale, tolerance_pct):
    skip_if_native_engine_unavailable(engine)

    L = room[0]
    W = room[1]
    H = room[2]
    fmin = 20
    dx = 343/(fmax*ppw)
    root_dir = tmp_path
    sim_dir = root_dir/'cpu'
    model_file = root_dir/'model.json'
    material = 'sabine_02.h5'
    num_modes = 25

    offset = dx*dx_scale
    room = RoomModelBuilder(L, W, H)
    room.add_source('S1', [offset, offset, offset])
    room.add_receiver('R1', [W-offset, L-offset, H-offset])
    room.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.03), root_dir / material)

    sim_setup_3d(
        model_json_file=model_file,
        mat_folder=root_dir,
        mat_files_dict={
            'Ceiling': material,
            'Floor': material,
            'Walls': material,
        },
        diff_source=True,
        duration=3.75,
        fcc_flag=fcc,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        save_folder=sim_dir,
        save_folder_gpu=sim_dir if engine != 'python' else None,
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

    actual, measured = detect_room_modes(
        filename=None,
        sim_dir=sim_dir,
        fmin=fmin,
        fmax=fmax,
        num_modes=num_modes,
        plot=False,
        width=W,
        length=L,
        height=H,
    )

    actual = np.array(actual[:num_modes])
    nearest = np.array([find_nearest(measured, mode) for mode in actual])
    rel_error_pct = np.abs(actual-nearest)/actual*100
    assert np.max(rel_error_pct) <= tolerance_pct
