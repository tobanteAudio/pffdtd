# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

import click
import h5py
import numpy as np
from resampy import resample
from scipy import signal

from pffdtd.signals.wavfile import save_as_wav_files


def process_outputs(
    sim_dir: pathlib.Path,
    out_file: pathlib.Path,
    *,
    resample_Fs: float | None = None,
    fcut_lowcut: float = 10.0,
    order_lowcut: int = 4,
    fcut_lowpass: float | None = None,
    order_lowpass: int = 4,
    diff: bool = False,
):
    with h5py.File(sim_dir / 'constants.h5', 'r') as constants:
        fs = float(constants['fs'][...])
        if not fcut_lowpass:
            fcut_lowpass = float(constants['fmax'][...])

    Ts = 1/fs

    with h5py.File(out_file, 'r') as h5f:
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

    sos = signal.butter(order_lowcut, fcut_lowcut, fs=fs, btype='high', output='sos')
    out = signal.sosfilt(sos, out)

    sos = signal.butter(order_lowpass, fcut_lowpass, fs=fs, btype='low', output='sos')
    out = signal.sosfilt(sos, out)

    out_f = resample(out, fs, resample_Fs, filter='kaiser_best')
    save_as_wav_files(out_f, resample_Fs, sim_dir, True)


@click.command(name='process-outputs', help='Process raw simulation output.')
@click.option('--diff', is_flag=True)
@click.option('--fmin', default=20.0)
@click.option('--sim_dir', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('out_file', nargs=1,  type=click.Path(exists=True, path_type=pathlib.Path))
def main(fmin, diff, sim_dir, out_file) -> None:
    process_outputs(
        sim_dir,
        out_file,
        resample_Fs=48000,
        fcut_lowcut=fmin,
        diff=diff,
    )
