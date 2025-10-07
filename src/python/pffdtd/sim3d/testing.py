# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import os
import pathlib
import subprocess

import pytest

from pffdtd.sim3d.engine import EnginePython3D

SIM3D_ENGINES = ['python', 'cpu']


def run_engine(sim_dir, engine):
    if engine == 'python':
        eng = EnginePython3D(sim_dir)
        eng.run_all(1)
        eng.save_outputs()
    else:
        assert engine in ['cpu', 'cuda', 'metal', 'sycl']

        exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
        assert exe.exists()
        assert exe.is_file()

        result = subprocess.run(
            args=[str(exe), 'sim3d', '-e', engine, '-p', '64', '-s', sim_dir],
            capture_output=True,
            check=True,
        )
        assert result.returncode == 0


def skip_if_native_engine_unavailable(engine):
    if engine in ['cpu', 'cuda', 'metal', 'sycl']:
        if not os.environ.get('PFFDTD_ENGINE_EXE'):
            pytest.skip('Native engine not available')
