# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.geometry.math import to_ixy
from pffdtd.sim2d.setup import sim_setup_2d


def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
    inx = 2
    iny = 2
    out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
    assert in_mask[inx, iny]
    return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy


sim_setup_2d(
    sim_dir='../../sim_data/Modes2D/cpu',
    room=(3.65, 6),
    Tc=20,
    rh=50,
    fmax=10_000.0,
    ppw=10.5,
    duration=10.0,
    refl_coeff=0.99,
    model_factory=model,
    apply_loss=True,
    diff=False,
    image=True,
    verbose=True,
)
