# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pathlib import Path

import cv2
import h5py
import numba as nb
import numpy as np
from tqdm import tqdm


class Engine2D:
    def __init__(self, sim_dir, out='out.h5', video=False):
        self.sim_dir = Path(sim_dir)
        self.video = video
        self.output_file = out

        with h5py.File(self.sim_dir / 'sim.h5', 'r') as h5f:
            self.fps = h5f['video_fps'][()]
            self.loss_factor = h5f['loss_factor'][()]
            self.Nt = h5f['Nt'][()]
            self.Nx = h5f['Nx'][()]
            self.Ny = h5f['Ny'][()]
            self.adj_bn = h5f['adj_bn'][...]
            self.bn_ixy = h5f['bn_ixy'][...]
            self.in_mask = h5f['in_mask'][...]
            self.in_sigs = h5f['in_sigs'][...]
            self.in_ixy = h5f['in_ixy'][...]
            self.out_ixy = h5f['out_ixy'][...]

        print(self.in_mask.shape)

    def run(self):
        Nt = self.Nt
        Nx = self.Nx
        Ny = self.Ny
        bn_ixy = self.bn_ixy
        adj_bn = self.adj_bn
        in_mask = self.in_mask
        in_sigs = self.in_sigs
        in_ixy = self.in_ixy
        out_ixy = self.out_ixy
        loss_factor = self.loss_factor
        fps = self.fps

        print('--SIM-ENGINE: Allocate python memory')
        u0 = np.zeros((Nx, Ny), dtype=np.float64)
        u1 = np.zeros((Nx, Ny), dtype=np.float64)
        u2 = np.zeros((Nx, Ny), dtype=np.float64)
        self.out = np.zeros((out_ixy.shape[0], Nt), dtype=np.float64)

        print(f"{self.out.shape}")
        print(f"{out_ixy.shape}")
        print(f"{u0.shape}")

        if self.video:
            video_name = self.sim_dir/'out.avi'
            print(f'--SIM-ENGINE: Create python video file: {video_name}')
            height, width = 1000, 1000  # u0.shape
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video = cv2.VideoWriter(video_name, fourcc, fps,
                                    (width, height), isColor=False)

        print('--SIM-ENGINE: Run python simulation')
        for nt in tqdm(range(Nt)):
            stencil_air(u0, u1, u2, in_mask)
            stencil_boundary_rigid(u0, u1, u2, bn_ixy, adj_bn)
            stencil_boundary_loss(u0, u2, bn_ixy, adj_bn, loss_factor)

            u0.flat[in_ixy] += in_sigs[nt]
            self.out[:, nt] = u0.flat[[out_ixy]]

            u2 = u1.copy()
            u1 = u0.copy()

            if self.video:
                img = np.abs(u0)
                img = cv2.normalize(img, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
                img.flat[bn_ixy] = 255
                img.flat[~in_mask] = 255
                img = cv2.resize(img, (1000, 1000))
                video.write(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

        if self.video:
            video.release()

        print(f"last: u0={u0.flat[in_ixy]} u1={u1.flat[in_ixy]} u2={u2.flat[in_ixy]}")

    def save_output(self):
        with h5py.File(self.sim_dir / self.output_file, 'w') as h5f:
            h5f.create_dataset('out', data=self.out)


@nb.njit(parallel=True)
def stencil_air(u0, u1, u2, mask):
    Nx, Ny = u1.shape
    for ix in nb.prange(1, Nx-1):
        for iy in range(1, Ny-1):
            if mask[ix*Ny+iy]:
                left = u1[ix-1, iy]
                right = u1[ix+1, iy]
                bottom = u1[ix, iy-1]
                top = u1[ix, iy+1]
                last = u2[ix, iy]
                u0[ix, iy] = 0.5 * (left+right+bottom+top) - last


@nb.njit(parallel=True)
def stencil_boundary_rigid(u0, u1, u2, bn_ixy, adj_bn):
    _, Ny = u1.shape
    Nb = bn_ixy.size
    for i in nb.prange(Nb):
        ib = bn_ixy[i]
        K = adj_bn[i]

        last1 = u1.flat[ib]
        last2 = u2.flat[ib]

        left = u1.flat[ib-1]
        right = u1.flat[ib + 1]
        top = u1.flat[ib + Ny]
        bottom = u1.flat[ib - Ny]

        neighbors = left + right + top + bottom
        u0.flat[ib] = (2 - 0.5 * K) * last1 + 0.5 * neighbors - last2


@nb.njit(parallel=True)
def stencil_boundary_loss(u0, u2, bn_ixy, adj_bn, loss_factor):
    Nb = bn_ixy.size
    for i in nb.prange(Nb):
        ib = bn_ixy[i]
        K = adj_bn[i]
        lf = loss_factor
        prev = u2.flat[ib]
        current = u0.flat[ib]

        u0.flat[ib] = (current + lf * (4 - K) * prev) / (1 + lf * (4 - K))
