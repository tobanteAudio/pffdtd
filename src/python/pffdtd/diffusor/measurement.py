# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pffdtd.common.wavfile import collect_wav_files, load_wav_files


def polar_response(y: np.array, fs: float):
    octave_bands = [
        # (63, 125),
        # (125, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
    ]

    bands = []
    for lowcut, highcut in octave_bands:
        sos = signal.butter(8, [lowcut, highcut], btype='band', output='sos', fs=fs)
        band = signal.sosfilt(sos, y, axis=-1)
        label = f'{lowcut}-{highcut} Hz'
        bands.append((np.sqrt(np.mean(band**2, axis=1)), label))

    scale = np.max([np.max(y) for y, _ in bands])
    norm = []
    for band in bands:
        norm.append((band[0] / scale * 100, band[1]))

    return norm


@click.command(name='measurement', help='Measure polar response.')
@click.argument('sim_dir', nargs=1,  type=click.Path(exists=True))
def main(sim_dir):
    sim_dir = Path(sim_dir)
    files = collect_wav_files(sim_dir, '*_out_normalised.wav')
    fs, out = load_wav_files(files)

    mic_angles = np.array(list(range(30, 152, 2)))

    constants = h5py.File(sim_dir / 'constants.h5', 'r')
    fmax = float(constants['fmax'][...])
    trim_ms = 10.5
    trim_samples = int(fs/1000*trim_ms)

    print(len(files))
    print(f"{fs=:.3f} Hz")
    print(f"{fmax=}")
    print(f"{trim_ms=}")
    print(f"{trim_samples=}")
    print(f"len={out.shape[-1]/fs:.2f} s")
    print(f"{out.shape=}")

    out = out[:, trim_samples:]
    times: np.ndarray = np.linspace(0.0, out.shape[-1]/fs, out.shape[-1])

    plt.plot(times, out[0, :])
    # plt.plot(times, out[45, :], label=f'{45}deg')
    plt.plot(times, out[-1, :])
    plt.grid(which='both')
    # plt.legend()
    plt.show()

    def _plot(ax, rms, title):
        ax.plot(np.deg2rad(mic_angles), rms)
        ax.set_title(title)
        ax.set_ylim((0.0, 100.0))
        ax.set_thetamin(0)
        ax.set_thetamax(180)

    rms_values = polar_response(out, fs)

    fig, ax = plt.subplots(3, 2, constrained_layout=True, subplot_kw={'projection': 'polar'})
    fig.suptitle('Diffusion')

    _plot(ax[0][0], rms_values[0][0], rms_values[0][1])
    _plot(ax[0][1], rms_values[1][0], rms_values[1][1])

    _plot(ax[1][0], rms_values[2][0], rms_values[2][1])
    _plot(ax[1][1], rms_values[3][0], rms_values[3][1])

    _plot(ax[2][0], rms_values[4][0], rms_values[4][1])
    _plot(ax[2][1], rms_values[5][0], rms_values[5][1])

    # _plot(ax[3][0], rms_values[6][0], rms_values[6][1])
    # _plot(ax[3][1], rms_values[7][0], rms_values[7][1])

    plt.show()
