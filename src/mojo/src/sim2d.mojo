# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from math import ceildiv
from memory import UnsafePointer
from pathlib import Path
from python import Python
from sys import argv, has_accelerator

from utils.index import Index

alias float_type = DType.float32
alias index_type = DType.int32


def main():
    constrained[has_accelerator(), "sim2d requires a supported accelerator"]()

    h5py = Python.import_module("h5py")
    np = Python.import_module("numpy")
    tqdm = Python.import_module("tqdm")

    args = argv()
    sim_dir = Path(args[1])
    debug_assert(sim_dir.is_dir())

    sim_file = sim_dir / "sim.h5"
    constants = sim_dir / "constants.h5"
    debug_assert(sim_file.is_file())
    debug_assert(constants.is_file())

    print(sim_file)
    print(constants)

    h5f = h5py.File(String(sim_file), "r")
    var fps = Float32(h5f["video_fps"][Python.tuple()])
    var loss_factor = Float32(h5f["loss_factor"][Python.tuple()])
    var Nt = Int(h5f["Nt"][Python.tuple()])
    var Nx = Int(h5f["Nx"][Python.tuple()])
    var Ny = Int(h5f["Ny"][Python.tuple()])
    var Npts = Nx * Ny
    var adj_bn = h5f["adj_bn"][:].astype(np.int32)
    var bn_ixy = h5f["bn_ixy"][:].astype(np.int32)
    var in_mask = h5f["in_mask"][:]
    var in_sigs = h5f["in_sigs"][:].astype(np.float32)
    var in_ixy = h5f["in_ixy"][:].astype(np.int32)
    var out_ixy = h5f["out_ixy"][:].astype(np.int32)
    var Nb = Int(adj_bn.shape[0])
    var Nr = Int(out_ixy.shape[0])
    var Ns = Int(in_ixy.shape[0])
    h5f.close()

    print("- fps:", fps)
    print("- loss_factor:", loss_factor)
    print("- Nt:", Nt)
    print("- Nx:", Nx)
    print("- Ny:", Ny)

    print('- adj_bn("{}"): {}'.format(String(adj_bn.dtype), String(adj_bn.shape)))
    print('- bn_ixy("{}"): {}'.format(String(bn_ixy.dtype), String(bn_ixy.shape)))
    print('- in_mask("{}"): {}'.format(String(in_mask.dtype), String(in_mask.shape)))
    print('- in_sigs("{}"): {}'.format(String(in_sigs.dtype), String(in_sigs.shape)))
    print('- in_ixy("{}"): {}'.format(String(in_ixy.dtype), String(in_ixy.shape)))
    print('- out_ixy("{}"): {}'.format(String(out_ixy.dtype), String(out_ixy.shape)))

    ctx = DeviceContext()
    print("Running on:", ctx.api())

    u_layout = Layout.row_major(Nx, Ny)

    adj_bn_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Nb))
    bn_ixy_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Nb))
    in_mask_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Int(in_mask.shape[0])))
    in_sigs_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Nt))
    in_ixy_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Ns))
    out_ixy_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(Nr))
    out_layout = RuntimeLayout[Layout.row_major[2]()].row_major(Index(Nr, Nt))

    u0_buf = ctx.enqueue_create_buffer[float_type](u_layout.size()).enqueue_fill(0)
    u1_buf = ctx.enqueue_create_buffer[float_type](u_layout.size()).enqueue_fill(0)
    u2_buf = ctx.enqueue_create_buffer[float_type](u_layout.size()).enqueue_fill(0)
    out_buf = ctx.enqueue_create_buffer[float_type](out_layout.size()).enqueue_fill(0)

    adj_bn_buf = ctx.enqueue_create_buffer[index_type](adj_bn_layout.size())
    bn_ixy_buf = ctx.enqueue_create_buffer[index_type](bn_ixy_layout.size())
    in_mask_buf = ctx.enqueue_create_buffer[DType.uint8](in_mask_layout.size())
    in_sigs_buf = ctx.enqueue_create_buffer[float_type](in_sigs_layout.size())
    in_ixy_buf = ctx.enqueue_create_buffer[index_type](in_ixy_layout.size())
    out_ixy_buf = ctx.enqueue_create_buffer[index_type](out_ixy_layout.size())
    ctx.synchronize()

    adj_bn_tensor = LayoutTensor[index_type, Layout.row_major[1]()](adj_bn_buf, adj_bn_layout)
    bn_ixy_tensor = LayoutTensor[index_type, Layout.row_major[1]()](bn_ixy_buf, bn_ixy_layout)
    in_mask_tensor = LayoutTensor[DType.uint8, Layout.row_major[1]()](in_mask_buf, in_mask_layout)
    in_sigs_tensor = LayoutTensor[float_type, Layout.row_major[1]()](in_sigs_buf, in_sigs_layout)
    in_ixy_tensor = LayoutTensor[index_type, Layout.row_major[1]()](in_ixy_buf, in_ixy_layout)
    out_ixy_tensor = LayoutTensor[index_type, Layout.row_major[1]()](out_ixy_buf, out_ixy_layout)
    out_tensor = LayoutTensor[float_type, Layout.row_major[2]()](out_buf, out_layout)

    adj_bn_ptr = adj_bn.ctypes.data.unsafe_get_as_pointer[index_type]()
    bn_ixy_ptr = bn_ixy.ctypes.data.unsafe_get_as_pointer[index_type]()
    in_mask_ptr = in_mask.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
    in_sigs_ptr = in_sigs.ctypes.data.unsafe_get_as_pointer[float_type]()
    in_ixy_ptr = in_ixy.ctypes.data.unsafe_get_as_pointer[index_type]()
    out_ixy_ptr = out_ixy.ctypes.data.unsafe_get_as_pointer[index_type]()

    ctx.enqueue_copy(adj_bn_buf, adj_bn_ptr)
    ctx.enqueue_copy(bn_ixy_buf, bn_ixy_ptr)
    ctx.enqueue_copy(in_mask_buf, in_mask_ptr)
    ctx.enqueue_copy(in_sigs_buf, in_sigs_ptr)
    ctx.enqueue_copy(in_ixy_buf, in_ixy_ptr)
    ctx.enqueue_copy(out_ixy_buf, out_ixy_ptr)
    ctx.synchronize()

    block_size = 16
    num_x_blocks = ceildiv(Nx, block_size)
    num_y_blocks = ceildiv(Ny, block_size)

    air_update_kernel = ctx.compile_function_checked[air_update, air_update]()
    boundary_rigid_update_kernel = ctx.compile_function_checked[boundary_rigid_update, boundary_rigid_update]()
    boundary_loss_update_kernel = ctx.compile_function_checked[boundary_loss_update, boundary_loss_update]()
    copy_input_signal_kernel = ctx.compile_function_checked[copy_input_signal, copy_input_signal]()
    copy_receiver_signal_kernel = ctx.compile_function_checked[copy_receiver_signal, copy_receiver_signal]()

    pbar_vox = tqdm.tqdm(total=Nt * Npts, desc="Voxel", unit="voxel", unit_scale=True, position=0, dynamic_ncols=True)
    pbar_frames = tqdm.tqdm(total=Nt, desc="Frame", unit="frame", unit_scale=True, position=1, dynamic_ncols=True)

    for n in range(Nt):
        ctx.enqueue_function_checked(
            air_update_kernel,
            u0_buf,
            u1_buf,
            u2_buf,
            in_mask_tensor,
            Nx,
            Ny,
            grid_dim=(num_x_blocks, num_y_blocks),
            block_dim=(block_size, block_size),
        )

        ctx.enqueue_function_checked(
            boundary_rigid_update_kernel,
            u0_buf,
            u1_buf,
            u2_buf,
            bn_ixy_tensor,
            adj_bn_tensor,
            Ny,
            grid_dim=ceildiv(Nb, block_size),
            block_dim=block_size,
        )

        ctx.enqueue_function_checked(
            boundary_loss_update_kernel,
            u0_buf,
            u2_buf,
            bn_ixy_tensor,
            adj_bn_tensor,
            loss_factor,
            grid_dim=ceildiv(Nb, block_size),
            block_dim=block_size,
        )

        ctx.enqueue_function_checked(
            copy_input_signal_kernel,
            u0_buf,
            in_ixy_tensor,
            in_sigs_tensor,
            n,
            grid_dim=ceildiv(Ns, block_size),
            block_dim=block_size,
        )

        ctx.enqueue_function_checked(
            copy_receiver_signal_kernel,
            u0_buf,
            out_ixy_tensor,
            out_tensor,
            n,
            grid_dim=ceildiv(Nr, block_size),
            block_dim=block_size,
        )

        ctx.synchronize()

        tmp = u2_buf
        u2_buf = u1_buf
        u1_buf = u0_buf
        u0_buf = tmp

        pbar_vox.update(Npts)
        pbar_frames.update(1)

    pbar_vox.close()
    pbar_frames.close()

    with out_buf.map_to_host() as host_buffer:
        var np_array = np.zeros(Python.tuple(Nr, Nt), dtype=np.float64)
        for r in range(Nr):
            for t in range(Nt):
                np_array[r, t] = host_buffer[r * Nt + t]

        h5f = h5py.File(String(sim_dir / "out.h5"), "w")
        h5f.create_dataset("out", data=np_array)
        h5f.close()


fn air_update(
    u0: UnsafePointer[Scalar[float_type]],
    u1: UnsafePointer[Scalar[float_type]],
    u2: UnsafePointer[Scalar[float_type]],
    in_mask: LayoutTensor[DType.uint8, Layout.row_major[1](), MutableAnyOrigin],
    Nx: Int,
    Ny: Int,
):
    var x = Int(global_idx.x) + 1
    var y = Int(global_idx.y) + 1
    var idx = x * Ny + y

    if (x < Nx - 1) and (y < Ny - 1) and (in_mask[idx][0] != 0):
        var left = u1[idx - 1]
        var right = u1[idx + 1]
        var bottom = u1[idx - Ny]
        var top = u1[idx + Ny]
        var last = u2[idx]

        u0[idx] = 0.5 * (left + right + bottom + top) - last


fn boundary_rigid_update(
    u0: UnsafePointer[Scalar[float_type]],
    u1: UnsafePointer[Scalar[float_type]],
    u2: UnsafePointer[Scalar[float_type]],
    bn_ixy: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    adj_bn: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    Ny: Int,
):
    var idx = Int(global_idx.x)
    if idx < bn_ixy.dim[0]():
        var ib = bn_ixy[idx][0]
        var K = Float32(adj_bn[idx][0])

        var last1 = u1[ib]
        var last2 = u2[ib]

        var left = u1[ib - 1]
        var right = u1[ib + 1]
        var bottom = u1[ib - Ny]
        var top = u1[ib + Ny]
        var neighbors = left + right + top + bottom

        u0[ib] = (2.0 - 0.5 * K) * last1 + 0.5 * neighbors - last2


fn boundary_loss_update(
    u0: UnsafePointer[Scalar[float_type]],
    u2: UnsafePointer[Scalar[float_type]],
    bn_ixy: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    adj_bn: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    loss_factor: Float32,
):
    var idx = Int(global_idx.x)
    if idx < bn_ixy.dim[0]():
        var ib = bn_ixy[idx][0]
        var current = u0[ib]
        var prev = u2[ib]
        var k4 = Float32(4 - adj_bn[idx][0])

        u0[ib] = (current + loss_factor * k4 * prev) / (1.0 + loss_factor * k4)


fn copy_input_signal(
    u0: UnsafePointer[Scalar[float_type]],
    in_ixy: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    in_sigs: LayoutTensor[float_type, Layout.row_major[1](), MutableAnyOrigin],
    n: Int,
):
    var source = Int(global_idx.x)
    if source < in_ixy.dim[0]():
        u0[in_ixy[source]] += in_sigs[n][0]


fn copy_receiver_signal(
    u0: UnsafePointer[Scalar[float_type]],
    out_ixy: LayoutTensor[index_type, Layout.row_major[1](), MutableAnyOrigin],
    output: LayoutTensor[float_type, Layout.row_major[2](), MutableAnyOrigin],
    n: Int,
):
    var receiver = Int(global_idx.x)
    if receiver < out_ixy.dim[0]():
        output[receiver, n] = u0[out_ixy[receiver]]
