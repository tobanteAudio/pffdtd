# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from pffdtd.analysis.waterfall import plot_waterfall
from pffdtd.common.wavfile import wavread
from pffdtd.filters.group_delay import group_delay_seconds, excess_group_delay_seconds


def plot_impulse_response_summary(x: np.ndarray, fs: float, *, fmax: float | None = None):
    if not fmax:
        fmax = fs/2

    n = x.shape[-1]
    nfft = n*4
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    H = np.fft.rfft(x, nfft)

    mag = np.maximum(np.abs(H), 1e-9)
    mag_dB = 20*np.log10(mag)
    mag_dB = mag_dB - np.max(mag_dB) + 85

    fig, axs = plt.subplots(3, 2)

    impulse_plot: Axes = axs[0][0]
    impulse_plot.plot(np.linspace(0.0, n/fs, n), x)
    impulse_plot.set_xlabel('Frequency [Hz]')
    impulse_plot.set_ylabel('Amplitude [FS]')
    impulse_plot.set_title('Impulse Response')
    impulse_plot.grid(which='both')

    plot_waterfall(x, fs, ax=axs[0][1])

    mag_plot: Axes = axs[1][0]
    mag_plot.semilogx(freqs, mag_dB)
    mag_plot.set_xlim(10, fmax)
    mag_plot.set_ylim(30, 90)
    mag_plot.set_xlabel('Frequency [Hz]')
    mag_plot.set_ylabel('Magnitude [dB]')
    mag_plot.set_title('Magnitude Response')
    mag_plot.grid(which='both')

    phase_plot: Axes = axs[2][0]
    phase_plot.semilogx(freqs, np.rad2deg(np.angle(H)))
    phase_plot.set_xlim(10, fmax)
    phase_plot.set_xlabel('Frequency [Hz]')
    phase_plot.set_ylabel('Phase [deg]')
    phase_plot.set_title('Phase Response')
    phase_plot.grid(which='both')

    group_delay_plot: Axes = axs[1][1]
    group_delay_plot.semilogx(freqs, group_delay_seconds(H, freqs)*1000)
    group_delay_plot.set_xlim(10, fmax)
    group_delay_plot.set_xlabel('Frequency [Hz]')
    group_delay_plot.set_ylabel('Group Delay [ms]')
    group_delay_plot.set_title('Group Delay')
    group_delay_plot.grid(which='both')

    excess_group_delay_plot: Axes = axs[2][1]
    excess_group_delay_plot.semilogx(freqs, excess_group_delay_seconds(H, freqs)*1000)
    excess_group_delay_plot.set_xlim(10, fmax)
    excess_group_delay_plot.set_xlabel('Frequency [Hz]')
    excess_group_delay_plot.set_ylabel('Group Delay [ms]')
    excess_group_delay_plot.set_title('Excess Group Delay')
    excess_group_delay_plot.grid(which='both')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2, left=0.05, right=0.957, top=0.922, bottom=0.1)

    return fig, axs


@click.command(name='summary', help='Quick display of IRs.')
@click.argument('impulse_path', nargs=1, type=click.Path(exists=True))
@click.option('--fmax', default=None, type=float)
def main(impulse_path, fmax):
    fs, ir = wavread(impulse_path)
    plot_impulse_response_summary(ir, fs, fmax=fmax)
    plt.show()
