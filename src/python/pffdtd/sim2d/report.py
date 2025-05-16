# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pffdtd.analysis.room_modes import room_modes


@click.command(name='report', help='Generate report.')
@click.option('--sim_dir', type=click.Path(exists=True))
@click.argument('out_file', nargs=-1, type=click.Path(exists=True))
def main(sim_dir, out_file) -> None:
    out_file = out_file[0]
    sim_dir = Path(sim_dir)

    with h5py.File(sim_dir / 'constants.h5', 'r') as constants:
        fs = float(constants['fs'][...])
        fmax = float(constants['fmax'][...])

    Ts = 1/fs
    fmin = 20.0
    trim_ms = 20
    trim_samples = int(fs/1000*trim_ms)

    with h5py.File(out_file, 'r') as file:
        out = file['out'][...]

    print(f"{out_file=}")
    print(f"{fs=:.3f} Hz")
    print(f"len={out.shape[-1]/fs:.2f} s")
    print(f"{trim_ms=}")
    print(f"{trim_samples=}")
    print(f"{out.shape=}")

    b = Ts/2*np.array([1, 1])
    a = np.array([1, -1])
    out = signal.lfilter(b, a, out)

    out /= np.max(np.abs(out))

    sos = signal.butter(4, fmin, fs=fs, btype='high', output='sos')
    out = signal.sosfilt(sos, out)

    sos = signal.butter(4, fmax, fs=fs, btype='low', output='sos')
    out = signal.sosfilt(sos, out)

    # out *= signal.windows.hann(out.shape[-1])
    spectrum: np.ndarray = np.fft.rfft(out, axis=-1)
    frequencies: np.ndarray = np.fft.rfftfreq(out.shape[-1], 1/fs)
    times: np.ndarray = np.linspace(0.0, out.shape[-1]/fs, out.shape[-1])

    dB: np.ndarray = 20*np.log10(np.abs(spectrum)+np.spacing(1))
    dB = dB-np.max(dB)

    print(times.shape)

    modes = room_modes(3.65, 6, 3)[:30]
    modes_f = [mode['frequency'] for mode in modes]

    plt.plot(times, out.squeeze(), label=f'{15}deg')
    plt.grid(which='both')
    plt.legend()
    plt.show()

    plt.semilogx(frequencies, dB.squeeze(), label=f'{15}deg')
    plt.vlines(modes_f, -60, 0.0, colors='r', linestyles='--')
    plt.xlim(10, 500)
    plt.ylim(-80, 0)
    plt.grid(which='both')
    plt.legend()
    plt.show()
