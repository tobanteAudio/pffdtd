# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from math import ceildiv
from sys import argv, has_accelerator

from utils.index import Index

alias float_dtype = DType.float32


def main():
    constrained[has_accelerator(), "This example requires a supported GPU"]()

    var args = argv()
    var vector_width = Int(args[1])
    var block_size = 5

    # Get context for the attached GPU
    var ctx = DeviceContext()

    # Allocate data on the GPU address space
    var rt_layout = RuntimeLayout[Layout.row_major[1]()].row_major(Index(vector_width))
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](rt_layout.size())
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](rt_layout.size())
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](rt_layout.size())

    # Fill in values across the entire width
    _ = lhs_buffer.enqueue_fill(1.25)
    _ = rhs_buffer.enqueue_fill(2.5)

    # Wrap the device buffers in tensors
    var lhs_tensor = LayoutTensor[float_dtype, Layout.row_major[1]()](lhs_buffer, rt_layout)
    var rhs_tensor = LayoutTensor[float_dtype, Layout.row_major[1]()](rhs_buffer, rt_layout)
    var out_tensor = LayoutTensor[float_dtype, Layout.row_major[1]()](out_buffer, rt_layout)

    # Calculate the number of blocks needed to cover the vector
    var grid_dim = ceildiv(vector_width, block_size)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function_checked[vector_addition, vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        vector_width,
        grid_dim=grid_dim,
        block_dim=block_size,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, Layout.row_major[1]()](host_buffer, rt_layout)
        print("Resulting vector:", host_tensor)


fn vector_addition(
    lhs_tensor: LayoutTensor[float_dtype, Layout.row_major[1](), MutableAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, Layout.row_major[1](), MutableAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, Layout.row_major[1](), MutableAnyOrigin],
    size: Int,
):
    """The calculation to perform across the vector on the GPU."""
    var global_tid = global_idx.x
    if global_tid < UInt(size):
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]
