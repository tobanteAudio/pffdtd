# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path

import click
import h5py
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import oaconvolve

from pffdtd.signals.wavfile import wavread


def deconvolve(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = y.shape[-1] + s.shape[-1] - 1
    Y = np.fft.rfft(np.atleast_2d(y), n=n)
    S = np.fft.rfft(s, n=n)
    H = np.zeros_like(Y)
    I = np.where(np.abs(S) > 1e-7)
    H[:, I] = Y[:, I] / S[I]
    return np.squeeze(np.fft.irfft(H, n=n))


def deconvolve_sim_outputs(sim_dir: Path, *, plot=True):
    sim_dir = Path(sim_dir)

    with h5py.File(sim_dir / 'signals.h5', 'r') as f:
        stimulus = f['in_sig'][()]

    with h5py.File(sim_dir / 'sim_outs_processed.h5', 'r') as f:
        u_out = f['r_out_f'][()]
        fs = f['Fs_f'][()]

    # mls = generate_max_len_seq(8)
    ir = deconvolve(u_out, stimulus)
    ir_dB = 20*np.log10(np.abs(ir)+1e-9)
    ir_dB -= np.max(ir_dB)

    if plot:
        _, axs = plt.subplots(4, 1)

        ax: Axes = axs[0]
        ax.set_title('MLS')
        ax.set_xlim(0.0, u_out.shape[-1]/fs)
        ax.plot(np.linspace(0.0, len(stimulus)/fs, len(stimulus)), stimulus)
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

        ax = axs[1]
        ax.set_title('Recording')
        ax.set_xlim(0.0, u_out.shape[-1]/fs)
        ax.plot(np.linspace(0.0, u_out.shape[-1]/fs, u_out.shape[-1]), u_out[0, :])
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

        ax = axs[2]
        ax.set_title('Impulse')
        ax.set_xlim(0.0, u_out.shape[-1]/fs)
        ax.plot(np.linspace(0.0, ir.shape[-1]/fs, ir.shape[-1]), ir[0, :])
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

        ax = axs[3]
        ax.set_title('Impulse [dB]')
        ax.set_xlim(0.0, u_out.shape[-1]/fs)
        ax.set_ylim(-105, +5)
        ax.plot(np.linspace(0.0, ir.shape[-1]/fs, ir.shape[-1]), ir_dB[0, :])
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

        plt.show()


@click.command(name='convolution', help='Convolution & deconvolution.')
@click.argument('impulse_path', nargs=-1, type=click.Path(exists=True))
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--stimulus', type=click.Path(exists=True))
def main(impulse_path, sim_dir, stimulus) -> None:
    if sim_dir:
        deconvolve_sim_outputs(sim_dir=sim_dir, plot=True)
        return

    fs_ir, ir = wavread(impulse_path)
    fs_sweep, sweep = wavread(stimulus)
    assert fs_ir == fs_sweep
    fs = fs_ir

    convolution = oaconvolve(sweep, ir)
    deconvolution = deconvolve(convolution, sweep)

    print(f'ir            = {ir.shape[-1]/fs:.2f} s')
    print(f'sweep         = {sweep.shape[-1]/fs:.2f} s')
    print(f'convolution   = {convolution.shape[-1]/fs:.2f} s')
    print(f'deconvolution = {deconvolution.shape[-1]/fs:.2f} s')

    _, axs = plt.subplots(4, 1)

    ax: Axes = axs[0]
    ax.set_title('IR')
    ax.set_xlim(0.0, len(ir)/fs)
    ax.plot(np.linspace(0.0, len(ir)/fs, len(ir)), ir)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    ax = axs[1]
    ax.set_title('Sweep')
    ax.set_xlim(0.0, len(convolution)/fs)
    ax.plot(np.linspace(0.0, len(sweep)/fs, len(sweep)), sweep)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    ax = axs[2]
    ax.set_title('Convolution')
    ax.set_xlim(0.0, len(convolution)/fs)
    ax.plot(np.linspace(0.0, len(convolution)/fs, len(convolution)), convolution)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    ax = axs[3]
    ax.set_title('Deconvolution')
    ax.set_xlim(0.0, len(ir)/fs)
    ax.plot(np.linspace(0.0, len(deconvolution)/fs, len(deconvolution)), deconvolution)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    plt.show()
