# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

from pathlib import Path

import numpy as np
import h5py


class SimConstants:
    """Class to keep simulation constants mostly in one place, writes to HDF5
    """

    def __init__(self, Tc, rh, h=None, fs=None, fmax=None, PPW=None, fcc=False, verbose=True):
        # Tc is temperature, rh is relative humidity <- this gives c (speed of sound)
        assert Tc >= -20
        assert Tc <= 50
        assert rh <= 100
        assert rh >= 10
        c = 343.2*np.sqrt(Tc/20)

        assert (h is not None) or (fs is not None) or (
            fmax is not None and PPW is not None)

        if fcc:
            l2 = 1.0
            l = np.sqrt(l2)
            assert l <= 1.0  # of course true
        else:
            l2 = 1/3
            l = np.sqrt(l2)
            assert l <= np.sqrt(1/3)  # check with round-off errors

        # back off to remove nyquist mode
        l *= 0.999
        l2 = l*l

        if h is not None:
            Ts = h/c*l
            fs = 1/Ts
        elif fs is not None:
            Ts = 1/fs
            h = c*Ts/l
        elif fmax is not None and PPW is not None:
            h = c/(fmax*PPW)  # PPW is points per wavelength (on Cartesian grid)
            Ts = h/c*l
            fs = 1/Ts
        else:
            raise RuntimeError('Invalid combination of h, fs, fmax & PPW')

        self.h = h
        self.c = c
        self.Ts = Ts
        self.fs = fs
        self.fmax = fmax
        self.l = l
        self.l2 = l2
        self.fcc = fcc

        self.Tc = Tc
        self.rh = rh

        if verbose:
            self.print(f'c    = {c:.2f} m/s')
            self.print(f'Ts   = {Ts*1e3:.6f} ms / {Ts*1e6:.3f} us')
            self.print(f'fs   = {fs:.2f} Hz')
            self.print(f'fmax = {fmax:.2f} Hz')
            self.print(f'h    = {h:.5f} m / {h*1e3:.2f} mm')
            self.print(f'l    = {l}')
            self.print(f'l2   = {l2}')

    def print(self, fstring):
        print(f'--CONSTANTS: {fstring}')

    # save to HDF5 file
    def save(self, save_folder):
        c = self.c
        h = self.h
        Ts = self.Ts
        l = self.l
        l2 = self.l2
        fs = self.fs
        fmax = self.fmax
        fcc = self.fcc
        Tc = self.Tc
        rh = self.rh

        save_folder = Path(save_folder)
        self.print(f'{save_folder=}')
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        else:
            assert save_folder.is_dir()

        with h5py.File(save_folder / Path('constants.h5'), 'w') as h5f:
            h5f.create_dataset('c', data=np.float64(c))
            h5f.create_dataset('h', data=np.float64(h))
            h5f.create_dataset('Ts', data=np.float64(Ts))
            h5f.create_dataset('fs', data=np.float64(fs))
            h5f.create_dataset('fmax', data=np.float64(fmax))
            h5f.create_dataset('l', data=np.float64(l))
            h5f.create_dataset('l2', data=np.float64(l2))
            h5f.create_dataset('fcc_flag', data=np.int8(fcc))
            h5f.create_dataset('Tc', data=np.float64(Tc))
            h5f.create_dataset('rh', data=np.float64(rh))
