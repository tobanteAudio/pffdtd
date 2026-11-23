# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import os
import pathlib
import subprocess

from pffdtd.absorption.admittance import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.analysis.room_modes import detect_room_modes
from pffdtd.geometry.math import to_ixy
from pffdtd.sim2d.process_outputs import process_outputs as process_outputs_2d
from pffdtd.sim2d.setup import sim_setup_2d
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.process_outputs import process_outputs as process_outputs_3d
from pffdtd.sim3d.setup import sim_setup_3d
from pffdtd.sim3d.testing import run_engine


def main():
    root_dir = pathlib.Path('sim_data/Modes_2D_vs_3D')
    root_dir_2d = root_dir/'2d'
    root_dir_3d = root_dir/'3d'
    root_dir_2d.mkdir(parents=True, exist_ok=True)
    root_dir_3d.mkdir(parents=True, exist_ok=True)

    engine = 'cpu'
    fcc = False
    fmax = 1000
    ppw = 7.75 if fcc else 10.5
    dx_scale = 3 if fcc else 2
    plot = False
    Tc = 20
    rh = 50
    duration = 1.5

    # L = 3
    # W = 1.8
    # H = 0.66
    L = 1.5*2
    W = 2
    H = 0.5
    fmin = 20
    dx = 343/(fmax*ppw)
    sim_dir = root_dir_3d/'cpu'
    model_file = root_dir_3d/'model.json'
    material = 'mat.h5'
    num_modes = 25

    offset = dx*dx_scale
    room = RoomModelBuilder(L, W, H)
    room.add_source('S1', [offset, offset, offset])
    room.add_receiver('R1', [W-offset, L-offset, H-offset])
    room.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.03), root_dir_3d / material)

    sim_setup_3d(
        model_json_file=model_file,
        mat_folder=root_dir_3d,
        mat_files_dict={
            'Ceiling': material,
            'Floor': material,
            'Walls': material,
        },
        diff_source=True,
        duration=duration,
        fcc_flag=fcc,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        Tc=Tc,
        rh=rh,
        save_folder=sim_dir,
        save_folder_gpu=sim_dir,
        Nprocs=1,
    )

    run_engine(sim_dir=sim_dir, engine=engine, precision=64, capture_output=False)

    process_outputs_3d(
        sim_dir=sim_dir,
        resample_Fs=48_000,
        fcut_lowcut=fmin,
        order_lowcut=4,
        fcut_lowpass=fmax,
        order_lowpass=8,
        symmetric_lowpass=True,
        air_abs_filter='none',
        save_wav=True,
        plot_raw=False,
        plot=plot,
    )

    detect_room_modes(
        filename=None,
        sim_dir=sim_dir,
        fmin=fmin,
        fmax=fmax,
        num_modes=num_modes,
        plot=plot,
        width=W,
        length=L,
        height=H,
    )

    def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
        in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
        inx = 1
        iny = 1
        out_ixy = [to_ixy(Nx-3, Ny-3, Nx, Ny)]
        assert in_mask[inx, iny]
        return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy

    sim_setup_2d(
        sim_dir=root_dir_2d,
        room=(L, W),
        Tc=Tc,
        rh=rh,
        fmax=fmax,
        ppw=10.5,
        duration=duration,
        refl_coeff=0.99,
        model_factory=model,
        apply_loss=True,
        diff=False,
        image=True,
        verbose=True,
    )

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.exists()
    assert exe.is_file()

    result = subprocess.run(
        args=[str(exe), 'sim2d', '-p', '64', '-e', engine, '-s', str(root_dir_2d), '-o', 'out.h5'],
        capture_output=False,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    process_outputs_2d(
        root_dir_2d,
        root_dir_2d/'out.h5',
        resample_Fs=48000,
        fcut_lowcut=fmin,
        fcut_lowpass=fmax,
    )

    detect_room_modes(
        filename=root_dir_2d/'R001_out_normalised.wav',
        sim_dir=root_dir_2d,
        fmin=fmin,
        fmax=fmax,
        num_modes=num_modes,
        plot=plot,
        width=W,
        length=L,
        height=H,
        dimensions=2,
    )


if __name__ == '__main__':
    main()
