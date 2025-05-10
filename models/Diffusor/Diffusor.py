# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import numpy as np
from tqdm import tqdm

from pffdtd.geometry.math import to_ixy, point_on_circle
from pffdtd.diffusion.diffusor import diffusor_bandwidth, quadratic_residue_diffusor, primitive_root_diffusor
from pffdtd.sim2d.setup import sim_setup_2d


def quantize_point(pos, dx, ret_err=True, scale=False):
    def quantize(i):
        low = np.floor(i/dx)
        high = np.ceil(i/dx)
        err_low = np.fabs(i-low*dx)
        err_high = np.fabs(i-high*dx)
        err = err_low if err_low < err_high else err_high
        i_q = low if err_low < err_high else high
        if scale:
            return i_q*dx, err
        return i_q, err

    x, y = pos
    xq, yq = quantize(x), quantize(y)
    if ret_err:
        return xq, yq
    return xq[0], yq[0]


def add_diffusor(prime, well_width, max_depth, room, in_mask, X, Y, dx, c, verbose=False):
    print('--DIFFUSOR: Quantize diffusor')
    dim = (well_width, max_depth)
    width = well_width
    depth = max_depth
    fmin_t, fmax_t = diffusor_bandwidth(dim[0], dim[1], c=c)

    (width_q, werr_q), (depth_q, derr_q) = quantize_point(dim, dx)
    width_q *= dx
    depth_q *= dx
    print(dim, width_q, depth_q)
    fmin_q, fmax_q = diffusor_bandwidth(width_q, depth_q, c=c)

    radius = 5
    total_width = 5
    pos = (room[0]/2-total_width/2, room[1]/2-radius-depth)
    pos_q = quantize_point(pos, dx)
    n = int(total_width/width_q)

    if verbose:
        print(f"  {pos_q=}")
        print(f"  {width=:.4f}")
        print(f"  {depth=:.4f}")
        print(f"  {fmin_t=:.4f}")
        print(f"  {fmax_t=:.4f}")
        print(f"  {width_q=:.4f}")
        print(f"  {depth_q=:.4f}")
        print(f"  {fmin_q=:.4f}")
        print(f"  {fmax_q=:.4f}")
        print(f"  error_w={werr_q/width*100:.2f}%")
        print(f"  error_d={derr_q/depth*100:.2f}%")

    print('--DIFFUSOR: Locate diffusor')
    depths, _ = primitive_root_diffusor(prime, g=None, depth=depth_q)
    depths = quadratic_residue_diffusor(prime, depth_q)
    prime = depths.shape[0]
    for w in tqdm(range(n)):
        xs = (room[0]/2-total_width/2)+w*width_q
        xe = xs+width_q
        ys = room[1]/2-5-depth_q
        ye = ys+depths[w % prime]+0.05
        in_mask[(X >= xs) & (Y >= ys) & (X < xe) & (Y < ye)] = False

    return in_mask


def make_receiver_arc(count, center, radius, dx, Nx, Ny):
    x, y = center
    angles = np.linspace(0.0, 180.0, count, endpoint=True)
    out_ixy = []
    out_cart = []
    for i in range(angles.shape[0]):
        x, y = point_on_circle(center, radius, np.deg2rad(angles[i]))
        xq, yq = quantize_point((x, y), dx, ret_err=False)
        idx = to_ixy(xq, yq, Nx, Ny)
        out_ixy.append(idx)
        out_cart.append((xq, yq))
    return out_ixy, out_cart


def diffusor_model_factory(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True

    # source input position
    x_in, y_in = Lx*0.5, Ly*0.5
    inx = int(np.round(x_in / dx + 0.5) + 1)
    iny = int(np.round(y_in / dx + 0.5) + 1)
    assert in_mask[inx, iny]

    # Diffusor
    print('--DIFFUSOR: Generate diffusor')
    prime = 17
    depth = 0.35
    width = 0.048
    in_mask = add_diffusor(prime, width, depth, (Lx, Ly),
                           in_mask, X, Y, dx, 343.0, True)

    # Receiver linear index
    print('--DIFFUSOR: Generate receivers')
    arc_radius = 5.0
    arc_center = (x_in, y_in-arc_radius)
    out_ixy, _ = make_receiver_arc(180, arc_center, arc_radius, dx, Nx, Ny)

    return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy


sim_setup_2d(
    sim_dir='../../sim_data/Diffusor/cpu',
    room=(45, 45),
    Tc=20,
    rh=50,
    fmax=20_000,
    ppw=10.5,
    duration=0.085,
    refl_coeff=0.99,
    model_factory=diffusor_model_factory,
    apply_loss=True,
    diff=False,
    image=True,
    verbose=True,
)
