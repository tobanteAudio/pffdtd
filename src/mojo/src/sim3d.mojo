# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from layout import Layout, IntTuple, RuntimeLayout, RuntimeTuple
from pathlib import Path
from python import Python, PythonObject
from sys import argv, has_accelerator

from utils.index import Index


@fieldwise_init
struct Constants[dtype: DType](ImplicitlyCopyable, Movable):
    alias float_type = Scalar[dtype]

    var l: Self.float_type
    var l2: Self.float_type
    var Ts: Self.float_type
    var fcc_flag: Int8

    @staticmethod
    def load(path: Path, debug_log: Bool = True) -> Self:
        h5py = Python.import_module("h5py")

        var file = h5py.File(String(path), "r")
        var l = Self.float_type(file["l"][Python.tuple()])
        var l2 = Self.float_type(file["l2"][Python.tuple()])
        var Ts = Self.float_type(file["Ts"][Python.tuple()])
        var fcc_flag = Int8(file["fcc_flag"][Python.tuple()])
        file.close()

        if debug_log:
            print("l        = {}".format(l))
            print("l2       = {}".format(l2))
            print("Ts       = {}".format(Ts))
            print("fcc_flag = {}".format(fcc_flag))

        return Constants[dtype](l, l2, Ts, fcc_flag)


def main():
    constrained[has_accelerator(), "sim3d requires a supported accelerator"]()

    var args = argv()
    var sim_dir = Path(args[1])
    var constants = Constants[DType.float32].load(sim_dir / "constants.h5")

    var layout = RuntimeLayout[Layout.row_major[3]()].row_major(Index(2, 2, 8))
    print(layout.idx2crd(RuntimeTuple(7)))
    print(layout.idx2crd(RuntimeTuple(8)))
    print(layout.idx2crd(RuntimeTuple(24)))

    print(layout(layout.idx2crd(RuntimeTuple(7))))
    print(layout(layout.idx2crd(RuntimeTuple(8))))
    print(layout(layout.idx2crd(RuntimeTuple(24))))
    # for x in range(layout.dim(0)):
    #     for y in range(layout.dim(1)):
    #         print(layout(RuntimeTuple(x, y)))
