# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""Function to set up a PFFDTD simulation with single source and multiple receivers
"""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Literal
import uuid

import click
import numpy as np

from pffdtd.common.misc import ensure_folder_exists
from pffdtd.sim3d.constants import SimConstants
from pffdtd.sim3d.materials import SimMaterials
from pffdtd.sim3d.room_geometry import RoomGeometry
from pffdtd.sim3d.rotate import rotate, sort_sim_data, copy_sim_data, fold_fcc_sim_data
from pffdtd.sim3d.signals import SimSignals
from pffdtd.voxelizer.cart_grid import CartGrid
from pffdtd.voxelizer.vox_grid import VoxGrid
from pffdtd.voxelizer.vox_scene import VoxScene


def sim_setup_3d(
    # The following are required but using None default so not positional
    insig_type=None,  # sig type (see sig_comms.py)
    fmax=None,  # fmax for simulation (to set grid spacing)
    PPW=None,  # points per wavelength (also to set grid spacing)
    save_folder=None,  # where to save .h5 files
    model_json_file=None,  # json export of model
    mat_folder=None,  # folder where to find .h5 DEF coefficients for wal impedances
    mat_files_dict=None,  # dict to link up materials to .h5 mat files
    duration=None,  # duration to simulate, in seconds

    # The following are not required
    Tc=20,  # temperature in deg C (sets sound speed)
    rh=50,  # relative humidity of air (configures air absorption post processing)
    source_num=1,  # 1-based indexing, source to simulate (in sources.csv)
    save_folder_gpu=None,  # folder to save gpu-prepared .h5 data (sorted and rotated and FCC-folded)
    draw_vox=False,  # draw voxelization
    draw_backend='mayavi',  # default, 'polyscope' better for larger grids
    diff_source=False,  # use this for single precision runs
    fcc_flag=False,  # to use FCC scheme
    bmin=None,  # to set custom scene bounds (useful for open scenes)
    bmax=None,  # to set custom scene bounds (useful for open scenes)
    Nvox_est=None,  # to manually set number of voxels (for ray-tri intersections) for voxelization
    Nh=None,  # to set voxel size in grid pacing (for ray-tri intersections)
    Nprocs=None,  # number of processes for multiprocessing, defaults to 80% of cores
    compress=None,  # GZIP compress for HDF5, 0 to 9 (fast to slow)
    rot_az_el=[0., 0.],  # to rotate the whole scene (including sources/receivers) -- to test robustness of scheme
    model_factory=None,
):
    assert Tc is not None
    assert rh is not None
    assert source_num > 0
    assert insig_type is not None
    assert fmax is not None
    assert PPW is not None
    assert save_folder is not None
    assert model_json_file is not None
    assert mat_folder is not None
    assert mat_files_dict is not None
    assert duration is not None

    # some constants for the simulation, in one place
    constants = SimConstants(Tc=Tc, rh=rh, fmax=fmax, PPW=PPW, fcc=fcc_flag)
    constants.save(save_folder)

    if (bmin is not None) and (bmax is not None):
        # custom bmin/bmax (for open scenes)
        bmin = np.array(bmin, dtype=np.float64)
        bmax = np.array(bmax, dtype=np.float64)

    if model_factory:
        model_factory(constants)

    # set up room geometry (reads in JSON export, rotates scene)
    room_geo = RoomGeometry(model_json_file, az_el=rot_az_el, bmin=bmin, bmax=bmax)
    room_geo.print_stats()

    # sources have to be specified in advance (edit JSON if necessary)
    Sxyz = room_geo.Sxyz[source_num-1]  # one source (one-based indexing)
    Rxyz = room_geo.Rxyz  # many receivers

    # link up the wall materials to impedance datasets
    materials = SimMaterials(save_folder=save_folder)
    materials.package(mat_files_dict=mat_files_dict,
                      mat_list=room_geo.mat_str, read_folder=mat_folder)

    # set the cartesian grid (also for FCC)
    cart_grid = CartGrid(h=constants.h, offset=3.5,
                         bmin=room_geo.bmin, bmax=room_geo.bmax, fcc=fcc_flag)
    cart_grid.print_stats()
    cart_grid.save(save_folder)

    # set up source/receiver positions and input signals
    sim_comms = SimSignals(save_folder=save_folder)  # reads from cart_grid
    sim_comms.prepare_source_pts(Sxyz)
    sim_comms.prepare_receiver_pts(Rxyz)
    sim_comms.prepare_source_signals(duration, sig_type=insig_type)
    if diff_source:
        sim_comms.diff_source()
    sim_comms.save(compress=compress)

    # set up the voxel grid (volume hierarchy for ray-triangle intersections)
    vox_grid = VoxGrid(room_geo, cart_grid, Nvox_est=Nvox_est, Nh=Nh)
    vox_grid.fill(Nprocs=Nprocs)
    vox_grid.print_stats()

    # 'voxelize' the scene (calculate FDTD mesh adjacencies and identify/correct boundary surfaces)
    vox_scene = VoxScene(room_geo, cart_grid, vox_grid, fcc=fcc_flag)
    vox_scene.calc_adj(Nprocs=Nprocs)
    vox_scene.check_adj_full()
    vox_scene.save(save_folder, compress=compress)

    # check that source/receivers don't intersect with boundaries
    sim_comms.check_for_clashes(vox_scene.bn_ixyz)

    # make copy for sorting/rotation for gpu
    if save_folder_gpu is not None and Path(save_folder_gpu) != Path(save_folder):
        copy_sim_data(save_folder, save_folder_gpu)
    if save_folder_gpu is not None:
        rotate(save_folder_gpu)
        if fcc_flag:
            fold_fcc_sim_data(save_folder_gpu)
        sort_sim_data(save_folder_gpu)

    # draw the voxelisation (use polyscope for dense grids)
    if draw_vox:
        room_geo.draw(wireframe=False, backend=draw_backend)
        vox_scene.draw(backend=draw_backend)
        room_geo.show(backend=draw_backend)


class Setup3D:
    duration: float
    fmax: float
    ppw: float
    fcc: bool
    Tc: float = 20
    rh: float = 50

    model_file: str
    bmin: list[float] | None = None
    bmax: list[float] | None = None
    rot_az_el: list[float] = [0.0, 0.0]
    materials: dict[str, str] = {}
    mat_folder: str | None = None

    source_index: int
    source_signal: Literal['impulse', 'impulse-highpass', 'hann10', 'hann20', 'hann5ms', 'dhann30']
    diff_source: bool = True

    compress: int = 0
    save_folder: str
    save_folder_gpu: str | None

    draw_vox: bool = True
    draw_backend: Literal['mayavi', 'polyscope'] = 'polyscope'


def run_setup3d_for_class(class_name):
    assert issubclass(class_name, Setup3D)

    sim = class_name()
    if sim.mat_folder:
        ensure_folder_exists(sim.mat_folder)

    if hasattr(sim, 'generate_materials'):
        sim.generate_materials()

    if hasattr(sim, 'generate_model'):
        def model_factory(c):
            return sim.generate_model(c)
    else:
        model_factory = None

    sim_setup_3d(
        insig_type=sim.source_signal,
        fmax=sim.fmax,
        PPW=sim.ppw,
        save_folder=sim.save_folder,
        model_json_file=sim.model_file,
        mat_folder=sim.mat_folder,
        mat_files_dict=sim.materials,
        duration=sim.duration,

        Tc=sim.Tc,
        rh=sim.rh,
        source_num=sim.source_index,
        save_folder_gpu=sim.save_folder_gpu,
        draw_vox=sim.draw_vox,
        draw_backend=sim.draw_backend,
        diff_source=sim.diff_source,
        fcc_flag=sim.fcc,
        bmin=sim.bmin,
        bmax=sim.bmax,
        Nvox_est=None,
        Nh=None,
        Nprocs=None if sys.platform.startswith('linux') else 1,
        compress=sim.compress,
        rot_az_el=sim.rot_az_el,
        model_factory=model_factory,
    )


def run_setup3d_for_file(sim_file):
    module_id = str(uuid.uuid1())
    spec = importlib.util.spec_from_file_location(module_id, sim_file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)

    for name, value in inspect.getmembers(loaded):
        if inspect.isclass(value) and issubclass(value, Setup3D) and name != 'Setup3D':
            run_setup3d_for_class(value)


@click.command(name='setup', help='Generate simulation files.')
@click.argument('sim_file', nargs=1, type=click.Path(exists=True))
def main(sim_file):
    run_setup3d_for_file(sim_file)
