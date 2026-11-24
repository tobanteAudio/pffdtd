# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import os
import pathlib
import subprocess

from click.testing import CliRunner
import h5py
import numpy as np
import pytest

from pffdtd.geometry.math import to_ixy
from pffdtd.cli import main as cli
from pffdtd.common.metal import metal_device_available
from pffdtd.sim2d.engine import pressure_to_grayscale
from pffdtd.sim2d.setup import generate_source_signal, sim_setup_2d


def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
    inx = 2
    iny = 2
    out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
    assert in_mask[inx, iny]
    return in_mask, [to_ixy(inx, iny, Nx, Ny)], out_ixy


def multi_source_model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
    in_ixy = [to_ixy(2, 2, Nx, Ny), to_ixy(2, 3, Nx, Ny)]
    out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
    assert in_mask.flat[in_ixy].all()
    return in_mask, in_ixy, out_ixy


def test_generate_unit_impulse():
    signal = generate_source_signal(128, 48_000, 2_000, band_limited=False)
    assert signal[0] == 1.0
    assert np.count_nonzero(signal) == 1


def test_generate_band_limited_impulse():
    sample_rate = 48_000
    cutoff = 2_000
    signal = generate_source_signal(4096, sample_rate, cutoff, band_limited=True)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, 1/sample_rate)

    passband = spectrum[np.argmin(np.abs(frequencies-1_000))]
    stopband = spectrum[np.argmin(np.abs(frequencies-4_000))]

    assert np.count_nonzero(signal) > 1
    assert stopband < passband*0.01


def test_band_limited_differentiated_source_is_bipolar(tmp_path):
    sim_setup_2d(
        sim_dir=tmp_path,
        room=(0.5, 0.5),
        Tc=20,
        rh=50,
        fmax=1_000,
        ppw=10.5,
        duration=0.01,
        refl_coeff=0.99,
        model_factory=model,
        apply_loss=True,
        diff=True,
        band_limited=True,
        image=False,
        verbose=False,
    )

    with h5py.File(tmp_path / 'sim.h5', 'r') as h5f:
        source = h5f['in_sigs'][...]

    assert np.min(source) < 0
    assert np.max(source) > 0


def test_pressure_to_grayscale_uses_symmetric_scale():
    pressure = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    image = pressure_to_grayscale(pressure, scale=1.0)
    assert np.array_equal(image, [[0, 0, 128, 255, 255]])


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 2D engine not available')
def test_sim2d_engines(tmp_path):
    room_size = 2.0
    sim_setup_2d(
        sim_dir=tmp_path,
        room=(room_size, room_size),
        Tc=20,
        rh=50,
        fmax=1000.0,
        ppw=10.5,
        duration=6.0,
        refl_coeff=0.991,
        model_factory=model,
        apply_loss=True,
        diff=True,
        image=True,
        verbose=True,
    )

    runner = CliRunner()
    args = ['sim2d', 'run', '--sim_dir', str(tmp_path), '--out', 'out-py.h5']
    result = runner.invoke(cli, args)
    assert result.exit_code == 0

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.exists()
    assert exe.is_file()

    result = subprocess.run(
        args=[str(exe), 'sim2d', '-p', '64', '-s', str(tmp_path), '-o', 'out-cpp.h5'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    with h5py.File(tmp_path / 'out-py.h5', 'r') as f:
        out_py = f['out'][...]

    with h5py.File(tmp_path / 'out-cpp.h5', 'r') as f:
        out_cpp = f['out'][...]

    with h5py.File(tmp_path / 'constants.h5', 'r') as f:
        c = float(f['c'][...])
        fs = float(f['fs'][...])

    assert not np.isnan(out_py).any()
    assert not np.isinf(out_py).any()
    assert not np.isnan(out_cpp).any()
    assert not np.isinf(out_cpp).any()
    assert np.allclose(out_py, out_cpp)

    expected_mode = c/(2*room_size)
    spectrum = np.abs(np.fft.rfft(out_cpp[0]*np.hanning(out_cpp.shape[-1])))
    frequencies = np.fft.rfftfreq(out_cpp.shape[-1], 1/fs)
    mode_band = (frequencies > 0.75*expected_mode) & (frequencies < 1.35*expected_mode)
    measured_mode = frequencies[mode_band][np.argmax(spectrum[mode_band])]
    assert measured_mode == pytest.approx(expected_mode, rel=0.01)


@pytest.mark.slow
@pytest.mark.skipif(not metal_device_available(), reason='No Metal device is available')
@pytest.mark.skipif(os.environ.get('PFFDTD_ENGINE_EXE') is None, reason='Native 2D engine not available')
def test_sim2d_metal_engine(tmp_path):
    sim_setup_2d(
        sim_dir=tmp_path,
        room=(0.5, 0.65),
        Tc=20,
        rh=50,
        fmax=1000.0,
        ppw=10.5,
        duration=0.025,
        refl_coeff=0.991,
        model_factory=multi_source_model,
        apply_loss=True,
        diff=True,
        image=False,
        verbose=False,
    )

    exe = pathlib.Path(os.environ.get('PFFDTD_ENGINE_EXE')).absolute()
    assert exe.is_file()

    for engine in ('cpu', 'metal'):
        result = subprocess.run(
            args=[str(exe), 'sim2d', '-e', engine, '-p', '32', '-s', str(tmp_path), '-o', f'out-{engine}.h5'],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0

    with h5py.File(tmp_path / 'out-cpu.h5', 'r') as f:
        out_cpu = f['out'][...]

    with h5py.File(tmp_path / 'out-metal.h5', 'r') as f:
        out_metal = f['out'][...]

    with h5py.File(tmp_path / 'sim.h5', 'r') as f:
        legacy_output_buffer_size = f['Nx'][()] * f['Ny'][()]

    assert out_metal.shape == out_cpu.shape
    required_output_buffer_size = out_metal.size
    # Ensure this test catches the old Metal allocation bug, which sized the output buffer as Nx*Ny instead of Nr*Nt.
    assert required_output_buffer_size > legacy_output_buffer_size
    assert np.isfinite(out_metal).all()

    peak = np.max(np.abs(out_cpu))
    assert peak > 0
    assert np.max(np.abs(out_metal - out_cpu)) < peak*5e-6
