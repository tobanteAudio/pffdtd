# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import pathlib

import click
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.signal import oaconvolve

from pffdtd.signals.level import normalize_to_RMS_dBFS
from pffdtd.signals.pink import generate_pink_noise
from pffdtd.signals.wavfile import wavread, wavwrite


def read_rew_rt60_txt(path: str | pathlib.Path) -> pd.DataFrame:
    """Reads an REW RT60 export as a pandas Dataframe"""
    df = pd.read_csv(path, sep=',', skiprows=10, nrows=24)
    for col in df.columns:
        df.rename(columns={col: col.strip()}, inplace=True)
    df.rename(columns={'Format is freq (Hz)': 'Frequency (Hz)'}, inplace=True)
    return df


def plot_rew_rt60_txt(col: str, a: pd.DataFrame, b: pd.DataFrame, a_label: str, b_label: str, ax_decay: Axes, ax_error: Axes):
    ax_decay.semilogx(a['Frequency (Hz)'], a[col], label=a_label)
    ax_decay.set_title(f'{col.strip(" (s)")}')
    ax_decay.semilogx(b['Frequency (Hz)'], b[col], label=b_label)
    ax_decay.set_xlim(50, 1600)
    ax_decay.set_ylim(0.1, 0.9)
    ax_decay.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax_decay.legend()

    ax_error.set_title(f'{col.strip(" (s)")} Error')
    ax_error.semilogx(b['Frequency (Hz)'], b[col]-a[col], label='B-A', color='red')
    ax_error.set_xlim(50, 1600)
    ax_error.set_ylim(-0.3, 0.3)
    ax_error.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax_error.legend()


@click.group(name='rew', help='Room EQ Wizard interop.')
def main():
    pass


@main.command(name='export', help='Export IR as REW offline measurement.')
@click.argument('impulse_path', nargs=1, type=click.Path(exists=True))
@click.option('--stimulus', type=click.Path(exists=True))
@click.option('--ref_dBC', 'ref_dBC', type=float, default=75.0)
def export(impulse_path, stimulus, ref_dBC):
    fs_ir, ir = wavread(impulse_path)
    ir = ir / np.sqrt(np.sum(ir**2))

    fs_sweep, sweep = wavread(stimulus)
    assert fs_ir == fs_sweep

    def _summary(x, name):
        dB_ref = 120.0
        peak = np.max(np.abs(x))
        rms = np.sqrt(np.mean(x**2))

        print(f'{name}:')
        print(f'  - Samples:  {len(x)}')
        print(f'  - Duration: {len(x)/fs_ir:.3f} s')
        print(f'  - Peak:     {20*np.log10(peak):.2f} dBFS')
        print(f'  - RMS:      {20*np.log10(rms):.2f} dBFS')
        print(f'  - Peak:     {20*np.log10(peak)+dB_ref:.2f} dBSPL')
        print(f'  - RMS:      {20*np.log10(rms)+dB_ref:.2f} dBSPL')

    pink = generate_pink_noise(10.0, fs_ir)
    level_ref = oaconvolve(pink, ir, mode='full')
    measurement = oaconvolve(sweep, ir, mode='full')
    measurement = normalize_to_RMS_dBFS(measurement, ref_dBC-120, x_ref=level_ref)

    _summary(ir, 'IR')
    _summary(sweep, 'Sweep')
    _summary(pink, 'Pink')
    _summary(level_ref, 'Level Ref')
    _summary(measurement, 'Measurement')

    wavwrite('out.wav', fs_ir, measurement)

    # plot_impulse_response_summary(ir, fs, fmax=fmax)
    # plt.show()


@main.command(name='rt60', help='Compare 1/3 octave RT60 exports.')
@click.argument('rt60_txt', nargs=2, type=click.Path(exists=True))
def rt60(rt60_txt):
    a_path = pathlib.Path(rt60_txt[0])
    b_path = pathlib.Path(rt60_txt[1])
    a = read_rew_rt60_txt(a_path)
    b = read_rew_rt60_txt(b_path)
    print(a)

    _, axs = plt.subplots(2, 2, constrained_layout=True)

    plot_rew_rt60_txt('T60M (s)', a, b, a_label=a_path.stem, b_label=b_path.stem, ax_decay=axs[0][0], ax_error=axs[1][0])
    plot_rew_rt60_txt('Topt (s)', a, b, a_label=a_path.stem, b_label=b_path.stem, ax_decay=axs[0][1], ax_error=axs[1][1])

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axs[0][0].xaxis.set_major_formatter(formatter)
    axs[1][0].xaxis.set_major_formatter(formatter)
    axs[0][1].xaxis.set_major_formatter(formatter)
    axs[1][1].xaxis.set_major_formatter(formatter)

    plt.show()
