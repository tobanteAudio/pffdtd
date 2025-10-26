# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from gpu.host import DeviceAttribute, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from sys import has_accelerator

from utils.index import Index


def main():
    constrained[has_accelerator(), "This example requires a supported accelerator"]()

    var num_devices = DeviceContext.number_of_devices()
    var num_cuda_devices = DeviceContext.number_of_devices(api="cuda")
    print("Num devices:       {}".format(num_devices))
    print("Num CUDA devices:  {}".format(num_cuda_devices))

    var ctx = DeviceContext()
    print("Running on:        {} via {}".format(ctx.name(), ctx.api()))
    print("GPU API version:   {}".format(ctx.get_api_version()))

    try:
        (free, total) = ctx.get_memory_info()

        var max_block_dim_x = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_X)
        var max_block_dim_y = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Y)
        var max_block_dim_z = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Z)
        var max_grid_dim_x = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
        var max_grid_dim_y = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Y)
        var max_grid_dim_z = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Z)
        var max_threads_per_block = ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
        var warp_size = ctx.get_attribute(DeviceAttribute.WARP_SIZE)

        print("Free memory:       {} MiB".format(free / UInt(1024**2)))
        print("Total memory:      {} MiB".format(total / UInt(1024**2)))
        print("Max block dim:     [{}, {}, {}]".format(max_block_dim_x, max_block_dim_y, max_block_dim_z))
        print("Max grid dim:      [{}, {}, {}]".format(max_grid_dim_x, max_grid_dim_y, max_grid_dim_z))
        print("Max threads/block: {}".format(max_threads_per_block))
        print("Warp size:         {}".format(warp_size))
    except:
        print("Failed to get memory information")

    alias layout_3d = Layout.row_major[3]()
    alias dtype = DType.float32

    var layout = RuntimeLayout[layout_3d].row_major(Index(1, 2, 4))
    var tensor_ptr = UnsafePointer[Scalar[dtype]].alloc(layout.size())
    var tensor = LayoutTensor[dtype, layout_3d](tensor_ptr, layout).fill(0.0)

    print(layout)
    print(layout.size())
    print(tensor)

    tensor_ptr.free()
