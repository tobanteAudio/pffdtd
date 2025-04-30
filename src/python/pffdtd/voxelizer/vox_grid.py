# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""
VoxGrid class for voxelizer. Uses multiprocessing.

Notes:
 - Performance will depend on geometry, grid spacing, voxel size and # processes
 - Simple *heuristic* default auto-tuning provided
 - Not designed for more than 10^6 voxels (initialisation becomes bottleneck)
 - Expected to run on a powerful CPU (+4 cores with SMT).
"""
import click
import numpy as np
from numpy import array as npa
from tqdm import tqdm

from pffdtd.common.misc import get_default_nprocs
from pffdtd.geometry.math import iceil
from pffdtd.sim3d.room_geometry import RoomGeometry
from pffdtd.voxelizer.cart_grid import CartGrid
from pffdtd.voxelizer.vox_grid_base import VoxGridBase, VoxBase


class Voxel(VoxBase):
    # using cubic voxels for simplicity
    def __init__(self, bmin, bmax, ixyz_start, Nhxyz, vox_idx):
        super().__init__(bmin, bmax)  # this has tri_idxs member
        # lower corner (of halo), greater than one
        self.ixyz_start = ixyz_start
        # size of voxel in grid-steps (not points), including halo
        self.Nhxyz = Nhxyz
        self.idx = vox_idx

# inherits draw_boxes() and and fill()


class VoxGrid(VoxGridBase):
    # only for voxelizer, only uses cubic voxels for now
    def __init__(self, room_geo, cart_grid, Nvox_est=None, Nh=None):
        super().__init__(room_geo)

        # tris = self.tris
        pts = self.pts
        # Npts = self.Npts
        Ntris = self.Ntris

        h = cart_grid.h
        xv = cart_grid.xv
        yv = cart_grid.yv
        zv = cart_grid.zv
        Nxyz = cart_grid.Nxyz
        Nx, Ny, Nz = Nxyz

        assert np.all(npa([xv[0], yv[0], zv[0]]) < np.amin(pts, axis=0))
        assert np.all(npa([xv[Nx-1], yv[Ny-1], zv[Nz-1]]) > np.amax(pts, axis=0))

        # Nh*h is width of non-overlapping part of voxel (with 0.5 spacing around points)
        # Nh is also min number of points along one dim
        # Nhx*h is size of voxel with halo (one extra layer)
        # Nhx is also number of points along one dim (Nxh>=Nh+2)

        if Nh is None and Nvox_est is None:  # heuristic, but seems ok
            fac = 0.025
            Nvox_est = iceil(fac*np.sqrt(Ntris * np.prod(Nxyz)))

        # calculate Nh if estimate given
        if Nvox_est is not None:
            assert Nh is None
            if Nvox_est == 0:
                raise

            if Nvox_est == 1:
                Nh = max((xv.size, yv.size, zv.size))-1
            elif Nvox_est > 1:
                vol = np.prod(room_geo.bmax-room_geo.bmin)
                vox_side = np.cbrt(vol/Nvox_est)
                Nh = int(np.round(vox_side/h))
                Nh = max(Nh, 4)

        assert Nh > 3  # not good to have too much overlap
        assert np.any(Nxyz >= Nh)
        self.print(f'Nh={Nh}')

        Nvox_xyz = np.int_(np.floor((Nxyz-2)/Nh))  # leaves room for shift to keep one-layer halo
        self.print(f'Nvox_xyz = {Nvox_xyz}, Nvox = {np.prod(Nvox_xyz)}')

        Nvox = int(np.prod(Nvox_xyz))  # keep as int so we can use in ranges

        vox_idx = 0
        self.timer.tic('allocate voxels')
        # allocate dummy voxels
        self.voxels = [Voxel(npa([0, 0, 0]), npa([np.inf, np.inf, np.inf]), npa([0, 0, 0]), npa([0, 0, 0]), 0) for i in range(Nvox)]
        self.print(self.timer.ftoc('allocate voxels'))

        self.timer.tic('initialise voxels')
        # Nh is step for voxels
        # Nvox_x, Nvox_y, Nvox_z = Nvox_xyz
        # can vectorize most of this but not creating Voxel objects
        vix = 0
        Nvx, Nvy, Nvz = Nvox_xyz

        pbar = tqdm(total=np.prod(Nvox_xyz), desc='vox grid init', ascii=True, leave=False, position=0)
        for vix in range(Nvx):
            ix_start = vix*Nh
            if vix < Nvx-1:
                ix_last = ix_start+Nh+1  # using matlab-style end (last)
            else:
                ix_last = Nx-1

            for viy in range(Nvy):
                iy_start = viy*Nh
                if viy < Nvy-1:
                    iy_last = iy_start+Nh+1
                else:
                    iy_last = Ny-1

                for viz in range(Nvz):
                    iz_start = viz*Nh
                    if viz < Nvz-1:
                        iz_last = iz_start+Nh+1
                    else:
                        iz_last = Nz-1

                    # box for voxel is one more layer thick
                    bmin = npa([xv[ix_start], yv[iy_start], zv[iz_start]])-0.5*h
                    bmax = npa([xv[ix_last], yv[iy_last], zv[iz_last]])+0.5*h  # using matlab-style end
                    ixyz_start = npa([ix_start, iy_start, iz_start])
                    ixyz_last = npa([ix_last, iy_last, iz_last])  # matlab-style end

                    vox = self.voxels[vox_idx]
                    vox.bmin = bmin
                    vox.bmax = bmax
                    vox.ixyz_start = ixyz_start
                    vox.Nhxyz = ixyz_last-ixyz_start+1
                    vox.idx = vox_idx

                    # start and end values of vox with one-layer still inbounds
                    assert np.all(vox.Nhxyz >= Nh+2)  # min size
                    assert np.all(vox.Nhxyz < 2*(Nh+2))  # max size
                    vox_idx += 1
                    pbar.update(1)
                    # print(f'{vix=},{viy=},{viz=}')

        pbar.close()
        self.print(self.timer.ftoc('initialise voxels'))
        assert vox_idx == Nvox

        self.Nvox_xyz = Nvox_xyz
        self.Nvox = Nvox
        self.Nh = Nh
        self.cg = cart_grid

    def print(self, fstring):
        print(f'--VOX_GRID: {fstring}')

    def print_stats(self):
        super().print_stats()


@click.command(name='grid')
@click.option('--az_el', nargs=2, type=float, default=[0, 0], help='two angles in deg')
@click.option('--draw/--no-draw', default=False)
@click.option('--draw-points/--no-draw-points', default=False, help='draw grid points')
@click.option('--h', type=float, required=True)
@click.option('--model_json', type=click.Path(exists=True), help='json file to import')
@click.option('--nh', type=int, default=None, help='Nh')
@click.option('--nvox_est', type=int, default=None, help='Nvox roughly')
@click.option('--nprocs', type=int, default=get_default_nprocs(), help='Number of processes')
@click.option('--offset', type=float, default=3.0)
def main(az_el, draw, draw_points, h, model_json, nh, nvox_est, nprocs, offset):
    assert nprocs > 0
    # assert nvox_est is not None or nh is not None
    assert h is not None
    assert model_json is not None

    room_geo = RoomGeometry(model_json, az_el=az_el)
    room_geo.print_stats()

    cart_grid = CartGrid(h, offset, room_geo.bmin, room_geo.bmax)
    cart_grid.print_stats()

    vox_grid = VoxGrid(room_geo, cart_grid, nvox_est, nh)
    vox_grid.fill(Nprocs=nprocs)
    vox_grid.print_stats()

    if draw:
        room_geo.draw()
        vox_grid.draw_boxes(tube_radius=cart_grid.h*vox_grid.Nh/100)
        print(f'{np.prod(cart_grid.Nxyz)=}')
        if draw_points:
            print('drawing grid points')
            cart_grid.draw_gridpoints()
        room_geo.show()
