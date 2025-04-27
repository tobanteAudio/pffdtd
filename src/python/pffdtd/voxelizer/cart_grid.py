# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

from pathlib import Path

import h5py
import numpy as np


class CartGrid():
    """This is a class for a Cartesian grid with grid spacing 'h' and some bounds.
    Saves minimal data in HDF5 file if needed outside.
    """

    def __init__(self, h=None, offset=None, bmin=None, bmax=None, fcc=False):
        assert h is not None
        assert offset is not None
        assert bmin is not None  # bbox min
        assert bmax is not None  # bbox max

        # ensure three-layer halo, for ABCs, etc.
        assert offset > 2.0  # distance offset

        xyzmin0 = bmin - offset*h
        xyzmax0 = bmax + offset*h

        # 64-bit ints signed (for simplicity and numpy indexing)
        Nx, Ny, Nz = np.int_(np.ceil((xyzmax0 - xyzmin0)/h))+1

        # make all dims even (so we can rotate any and halve)
        if fcc:
            Nx += (Nx % 2)
            Ny += (Ny % 2)
            Nz += (Nz % 2)
            self.print('To use FCC subgrid')

        # grid vectors
        xv, yv, zv = np.ogrid[0.:Nx, 0.:Ny, 0.:Nz]

        xv = xv.ravel()*h + xyzmin0[0]
        yv = yv.ravel()*h + xyzmin0[1]
        zv = zv.ravel()*h + xyzmin0[2]

        xyzmin = np.array([xv[0], yv[0], zv[0]])
        xyzmax = np.array([xv[-1], yv[-1], zv[-1]])

        assert np.all(xyzmin == xyzmin0)
        assert np.all(xyzmax >= xyzmax0)

        self.fcc = fcc
        self.h = h
        self.offset = offset
        self.xv = xv
        self.yv = yv
        self.zv = zv
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nxyz = np.array([Nx, Ny, Nz])
        self.Npts = np.prod(self.Nxyz)
        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        self.print_stats()

    def save(self, folder, filename='cart_grid.h5'):
        """Save to HDF5 file
        """
        xv = self.xv
        yv = self.yv
        zv = self.zv
        h = self.h

        folder = Path(folder)
        self.print(f'{folder=}')
        if not folder.exists():
            folder.mkdir(parents=True)
        else:
            assert folder.is_dir()

        compression = {'compression': 'gzip', 'compression_opts': 9}
        file = h5py.File(folder / filename, 'w')
        file.create_dataset('xv', data=xv, **compression)
        file.create_dataset('yv', data=yv, **compression)
        file.create_dataset('zv', data=zv, **compression)
        file.create_dataset('h', data=np.float64(h))
        file.close()

    def draw_gridpoints(self, backend='mayavi'):
        """Don't use this unless grid is small
        """
        xv = self.xv
        yv = self.yv
        zv = self.zv
        h = self.h

        X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
        if backend == 'mayavi':
            from mayavi import mlab
            mlab.points3d(
                X.flat[:], Y.flat[:], Z.flat[:],
                color=(0, 0, 0),
                mode='sphere',
                resolution=4,
                scale_factor=h/4
            )
            mlab.draw()
        else:
            raise NotImplementedError('TODO for polyscope')

    def print_stats(self):
        cg = self
        self.print(f'{cg.h=}')
        self.print(f'{cg.Nxyz=}')
        self.print(f'{cg.Npts=}')
        self.print(f'{cg.xyzmin=}')
        self.print(f'{cg.xyzmax=}')

        f32, f64 = self.memory_requirements()
        grid = 'fcc' if self.fcc else 'cart'
        self.print(f'{f32:.3f}GB in two-grid {grid} float32')
        self.print(f'{f64:.3f}GB in two-grid {grid} float64')

    def memory_requirements(self) -> tuple[float, float]:
        Npts = self.Npts
        Nbytes = Npts * 8 * 2
        NGbytes = Nbytes/1e9
        if self.fcc:
            return NGbytes/4, NGbytes/2
        return NGbytes/2, NGbytes

    def print(self, fstring):
        print(f'--CART_GRID: {fstring}')
