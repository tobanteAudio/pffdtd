# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

import click
import h5py
import numpy as np
from resampy import resample
from scipy import signal

from pffdtd.dsp.wavfile import save_as_wav_files


@click.command(name='process-outputs', help='Process raw simulation output.')
@click.option('--diff', is_flag=True)
@click.option('--fmin', default=20.0)
@click.option('--sim_dir', type=click.Path(exists=True))
@click.argument('out_file', nargs=1,  type=click.Path(exists=True))
def main(fmin, diff, sim_dir, out_file):
    sim_dir = pathlib.Path(sim_dir)

    constants = h5py.File(sim_dir / 'constants.h5', 'r')
    fs = float(constants['fs'][...])
    fmax = float(constants['fmax'][...])
    Ts = 1/fs

    h5f = h5py.File(out_file, 'r')
    out: np.ndarray = h5f['out'][...]

    print(f"{out_file=}")
    print(f"{fs=:.3f} Hz")
    print(f"len={out.shape[-1]/fs:.2f} s")
    print(f"{out.shape=}")

    if diff:
        b = Ts/2*np.array([1, 1])
        a = np.array([1, -1])
        out = signal.lfilter(b, a, out)

    out /= np.max(np.abs(out))

    sos = signal.butter(4, fmin, fs=fs, btype='high', output='sos')
    out = signal.sosfilt(sos, out)

    sos = signal.butter(4, fmax, fs=fs, btype='low', output='sos')
    out: np.ndarray = signal.sosfilt(sos, out)

    Fs_target = 48000
    out_f = resample(out, fs, Fs_target, filter='kaiser_best')
    save_as_wav_files(out_f, Fs_target, sim_dir, True)
