# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import os
import pathlib
import subprocess

import pytest

from pffdtd.common.metal import metal_device_available
from pffdtd.sim3d.engine import EnginePython3D


def engines_to_test() -> list[str]:
    engines = ['python', 'cpu']
    extra = os.environ.get('PFFDTD_EXTRA_TEST_ENGINES')
    if extra:
        for engine in extra.split(';'):
            # Metal is excluded, because it doesn't support float64
            assert engine in ['cuda', 'sycl']
            engines.append(engine)
    return engines


def run_engine(
    sim_dir: pathlib.Path,
    engine: str,
    precision: int | None = None,
    capture_output: bool = True,
):
    if engine == 'python':
        eng = EnginePython3D(sim_dir)
        eng.run_all(1)
        eng.save_outputs()
    else:
        assert engine in ['cpu', 'cuda', 'metal', 'sycl']

        exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
        assert exe.exists()
        assert exe.is_file()

        if not precision:
            precision = 32 if engine == 'metal' else 64

        if precision == 64 and engine == 'metal':
            raise ValueError('metal engine only supports precision "32"')

        result = subprocess.run(
            args=[str(exe), 'sim3d', '-e', engine, '-p', f'{precision}', '-s', str(sim_dir)],
            capture_output=capture_output,
            check=True,
        )
        assert result.returncode == 0


def skip_if_native_engine_unavailable(engine):
    if engine in ['cpu', 'cuda', 'metal', 'sycl']:
        if not os.environ.get('PFFDTD_ENGINE_EXE'):
            pytest.skip('Native engine not available')
    if engine == 'metal' and not metal_device_available():
        pytest.skip('No Metal device is available')


def skip_if_fcc_unsupported(engine):
    if engine in ['metal', 'sycl']:
        pytest.skip(f'"{engine}" engine does does not support FCC grids')
