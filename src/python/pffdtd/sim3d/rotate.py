# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""
For multi-GPU execution:
    - best to permute dimensions for descending order (last dim continguous)
    - indices all need to be sorted (and corresponding data reordered)
    - fold FCC subgrid onto itself here (fills half Cartesian grid)
"""

from pathlib import Path
import shutil

import h5py
import numpy as np

from pffdtd.common.timerdict import TimerDict
from pffdtd.geometry.math import ind2sub3d


def rotate(sim_dir, tr=None, compress=False):
    # NB: we keep cart_grid.h5 untouched and that has original Nx,Ny,Nz if needed
    def _print(fstring):
        print(f'--ROTATE_DATA: {fstring}')
    timer = TimerDict()
    sim_dir = Path(sim_dir)

    timer.tic('read')
    with h5py.File(sim_dir / Path('vox_out.h5'), 'r') as h5f:
        Nx = h5f['Nx'][()]
        Ny = h5f['Ny'][()]
        Nz = h5f['Nz'][()]

    if tr is None:
        tr = np.argsort(np.array([Nx, Ny, Nz]))[::-1]  # descending (Nx is non-contiguous -- want Ny*Nz min)
    else:
        assert np.all(np.sort(tr) == np.array([0, 1, 2]))
    _print(f'{tr=}')
    if np.all(tr == np.array([0, 1, 2])):
        _print('no rotate')
        _print(timer.ftoc('read'))
        return  # no op

    # read
    with h5py.File(sim_dir / Path('vox_out.h5'), 'r') as h5f:
        xv = h5f['xv'][()]
        yv = h5f['yv'][()]
        zv = h5f['zv'][()]
        adj_bn = h5f['adj_bn'][...]
        bn_ixyz = h5f['bn_ixyz'][...]

    NN = adj_bn.shape[1]
    if NN == 6:
        iVV = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    else:
        iVV = np.array([[+1, +1, 0], [-1, -1, 0], [0, +1, +1], [0, -1, -1], [+1, 0, +1], [-1, 0, -1],
                        [+1, -1, 0], [-1, +1, 0], [0, +1, -1], [0, -1, +1], [+1, 0, -1], [-1, 0, +1]])

    with h5py.File(sim_dir / Path('signals.h5'), 'r') as h5f:
        in_ixyz = h5f['in_ixyz'][...]
        out_ixyz = h5f['out_ixyz'][...]
        Nr = h5f['Nr'][()]
        Ns = h5f['Ns'][()]
        # Nt = h5f['Nt'][()]
    _print(timer.ftoc('read'))

    assert in_ixyz.shape[0] == Ns
    assert out_ixyz.shape[0] == Nr

    timer.tic('transpose')
    # swap and reorder

    def _swap3(a, b, c, tr):
        abcl = [a, b, c]
        return [abcl[i] for i in tr]  # swap with order

    Nxt, Nyt, Nzt = _swap3(Nx, Ny, Nz, tr)
    bn_ixyzt = np.array(_swap3(*ind2sub3d(bn_ixyz, Nx, Ny, Nz), tr)).T @ np.array([Nzt*Nyt, Nzt, 1])
    in_ixyzt = np.array(_swap3(*ind2sub3d(in_ixyz, Nx, Ny, Nz), tr)).T @ np.array([Nzt*Nyt, Nzt, 1])
    out_ixyzt = np.array(_swap3(*ind2sub3d(out_ixyz, Nx, Ny, Nz), tr)).T @ np.array([Nzt*Nyt, Nzt, 1])
    xvt, yvt, zvt = _swap3(xv, yv, zv, tr)
    assert xvt.size == Nxt
    assert yvt.size == Nyt
    assert zvt.size == Nzt
    _print(timer.ftoc('transpose'))

    timer.tic('reorder adj')
    # reorder adj_bn columns
    jj = np.zeros((NN,), dtype=np.int_)
    jj = np.array([np.flatnonzero(np.all(ivv[tr] == iVV, axis=-1))[0] for ivv in iVV])
    _print(f'{jj=}')
    ia = np.argsort(jj)
    adj_bnt = adj_bn[:, ia]
    timer.toc('reorder adj')

    timer.tic('write')
    if compress:
        kw = {'compression': 'gzip', 'compression_opts': 9}
    else:
        kw = {}
    # overwrite
    with h5py.File(sim_dir / Path('signals.h5'), 'r+') as h5f:
        h5f['in_ixyz'][...] = in_ixyzt
        h5f['out_ixyz'][...] = out_ixyzt

    with h5py.File(sim_dir / Path('vox_out.h5'), 'r+') as h5f:
        h5f['bn_ixyz'][...] = bn_ixyzt
        h5f['adj_bn'][...] = adj_bnt
        h5f['Nx'][()] = Nxt
        h5f['Ny'][()] = Nyt
        h5f['Nz'][()] = Nzt
        # these take different sizes, have to clobber
        del h5f['xv']
        h5f.create_dataset('xv', data=xvt, **kw)
        del h5f['yv']
        h5f.create_dataset('yv', data=yvt, **kw)
        del h5f['zv']
        h5f.create_dataset('zv', data=zvt, **kw)

    _print(timer.ftoc('write'))


def sort_sim_data(sim_dir):
    def _print(fstring):
        print(f'--SORT_DATA: {fstring}')
    timer = TimerDict()
    sim_dir = Path(sim_dir)

    timer.tic('read')
    # read
    with h5py.File(sim_dir / Path('vox_out.h5'), 'r') as h5f:
        adj_bn = h5f['adj_bn'][...]
        bn_ixyz = h5f['bn_ixyz'][...]
        mat_bn = h5f['mat_bn'][...]
        saf_bn = h5f['saf_bn'][...]

    with h5py.File(sim_dir / Path('signals.h5'), 'r') as h5f:
        in_ixyz = h5f['in_ixyz'][...]
        out_ixyz = h5f['out_ixyz'][...]
        out_alpha = h5f['out_alpha'][...]
        in_sigs = h5f['in_sigs'][...]

    _print(timer.ftoc('read'))

    timer.tic('reorder')
    # sort rows
    ii = np.argsort(bn_ixyz)
    bn_ixyz = bn_ixyz[ii]
    adj_bn = adj_bn[ii]
    mat_bn = mat_bn[ii]
    saf_bn = saf_bn[ii]

    ii = np.argsort(in_ixyz)
    in_ixyz = in_ixyz[ii]
    in_sigs = in_sigs[ii]

    ii = np.argsort(out_ixyz)
    out_ixyz = out_ixyz[ii]
    # out_alpha = out_alpha[ii] #will apply to reordered signals
    out_reorder = np.argsort(ii)
    _print(timer.ftoc('reorder'))

    timer.tic('write')
    # overwrite
    with h5py.File(sim_dir / Path('signals.h5'), 'r+') as h5f:
        h5f['in_ixyz'][...] = in_ixyz
        h5f['in_sigs'][...] = in_sigs
        h5f['out_ixyz'][...] = out_ixyz
        h5f['out_alpha'][...] = out_alpha
        h5f['out_reorder'][...] = out_reorder

    with h5py.File(sim_dir / Path('vox_out.h5'), 'r+') as h5f:
        h5f['bn_ixyz'][...] = bn_ixyz
        h5f['adj_bn'][...] = adj_bn
        h5f['mat_bn'][...] = mat_bn
        h5f['saf_bn'][...] = saf_bn

    _print(timer.ftoc('write'))


def fold_fcc_sim_data(sim_dir):
    def _print(fstring):
        print(f'--FOLD_FCC_DATA: {fstring}')

    timer = TimerDict()
    sim_dir = Path(sim_dir)
    with h5py.File(sim_dir / Path('vox_out.h5'), 'r') as h5f:
        Nx = h5f['Nx'][()]
        Ny = h5f['Ny'][()]
        Nz = h5f['Nz'][()]
    assert (Ny % 2) == 0

    with h5py.File(sim_dir / Path('vox_out.h5'), 'r') as h5f:
        adj_bn = h5f['adj_bn'][...]
        bn_ixyz = h5f['bn_ixyz'][...]

    with h5py.File(sim_dir / Path('signals.h5'), 'r') as h5f:
        in_ixyz = h5f['in_ixyz'][...]
        out_ixyz = h5f['out_ixyz'][...]

    with h5py.File(sim_dir / Path('constants.h5'), 'r') as h5f:
        fcc_flag = h5f['fcc_flag'][...]

    assert fcc_flag == 1

    Nyh = np.int_(Ny/2)+1

    bix, biy, biz = ind2sub3d(bn_ixyz, Nx, Ny, Nz)
    ii = biy >= Ny/2

    # bn_ixyz
    bn_ixyz[ii] = np.c_[bix[ii], Ny-biy[ii]-1, biz[ii]] @ np.array([Nz*Nyh, Nz, 1])
    bn_ixyz[~ii] = np.c_[bix[~ii], biy[~ii], biz[~ii]] @ np.array([Nz*Nyh, Nz, 1])

    adj_bn[ii, 0], adj_bn[ii, 6] = adj_bn[ii, 6], adj_bn[ii, 0]
    adj_bn[ii, 1], adj_bn[ii, 7] = adj_bn[ii, 7], adj_bn[ii, 1]
    adj_bn[ii, 2], adj_bn[ii, 9] = adj_bn[ii, 9], adj_bn[ii, 2]
    adj_bn[ii, 3], adj_bn[ii, 8] = adj_bn[ii, 8], adj_bn[ii, 3]

    # in_ixyz
    bix, biy, biz = ind2sub3d(in_ixyz, Nx, Ny, Nz)
    ii = biy >= Ny/2
    in_ixyz[ii] = np.c_[bix[ii], Ny-biy[ii]-1, biz[ii]] @ np.array([Nz*Nyh, Nz, 1])
    in_ixyz[~ii] = np.c_[bix[~ii], biy[~ii], biz[~ii]] @ np.array([Nz*Nyh, Nz, 1])

    # out_ixyz
    bix, biy, biz = ind2sub3d(out_ixyz, Nx, Ny, Nz)
    ii = biy >= Ny/2
    out_ixyz[ii] = np.c_[bix[ii], Ny-biy[ii]-1, biz[ii]] @ np.array([Nz*Nyh, Nz, 1])
    out_ixyz[~ii] = np.c_[bix[~ii], biy[~ii], biz[~ii]] @ np.array([Nz*Nyh, Nz, 1])

    timer.tic('write')
    # write
    with h5py.File(sim_dir / Path('signals.h5'), 'r+') as h5f:
        h5f['in_ixyz'][...] = in_ixyz
        h5f['out_ixyz'][...] = out_ixyz

    with h5py.File(sim_dir / Path('vox_out.h5'), 'r+') as h5f:
        h5f['bn_ixyz'][...] = bn_ixyz
        h5f['adj_bn'][...] = adj_bn
        h5f['Ny'][()] = Nyh

    with h5py.File(sim_dir / Path('constants.h5'), 'r+') as h5f:
        h5f['fcc_flag'][()] = 2

    _print(timer.ftoc('write'))


def copy_sim_data(src_sim_dir, dst_sim_dir):
    def _print(fstring):
        print(f'--COPY DATA: {fstring}')
    src_sim_dir = Path(src_sim_dir)
    _print(f'{src_sim_dir=}')
    assert src_sim_dir.is_dir()

    dst_sim_dir = Path(dst_sim_dir)
    _print(f'{dst_sim_dir=}')
    if not dst_sim_dir.exists():
        dst_sim_dir.mkdir(parents=True)
    else:
        assert dst_sim_dir.is_dir()
    for file in src_sim_dir.glob('*.h5'):
        _print(f'copying {file}')
        shutil.copy(file, dst_sim_dir)
