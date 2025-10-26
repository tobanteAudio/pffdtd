# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path
from python import Python, PythonObject
from sys import argv


def main():
    var raw_args = argv()
    var argparse = Python.import_module("argparse")
    var parser = argparse.ArgumentParser(raw_args[0])
    parser.add_argument("sim_dir", help="Simulation directory")

    var args: PythonObject
    try:
        args = parser.parse_args(Python.list([String(arg) for arg in raw_args])[1:])
    except:
        return

    var sim_dir = Path(String(args.sim_dir))
    print("Arguments:", args)
    print(sim_dir)
