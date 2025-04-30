# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy.signal import sosfilt

from pffdtd.common.plot import plot_styles
from pffdtd.common.wavfile import collect_wav_files, wavread
from pffdtd.dsp.octave import octave_bandpass


def clarity(x: np.ndarray, fs: float, early_ms: float) -> float:
    """The early to late energy ratio in dB, using sound energy in the first X ms as the 'early' part.

    - C50 is most often used as an indicator of speech clarity.
    - C80 is most often used as an indicator of music clarity.
    """
    e = x**2
    dt = 1/fs
    idx = int(round(early_ms/1000 * fs))
    E_early = e[:idx].sum() * dt
    E_late = e[idx:].sum() * dt
    return 10 * np.log10(E_early / E_late)


def energy_decay_curve(x) -> np.ndarray:
    edc = np.cumsum(x[::-1]**2)[::-1]
    edc_dB = 10 * np.log10(edc / np.max(edc))
    return edc_dB


def early_decay_time(edc_dB, fs):
    edc_dB -= np.max(edc_dB)
    end_idx = np.where(edc_dB <= -10)[0][0]

    t = np.arange(len(edc_dB)) / fs
    return t[end_idx]*6


def decay_time(edc_dB, fs, t20=False):
    threshold = -25 if t20 else -35
    multiplier = 3 if t20 else 2

    edc_dB -= np.max(edc_dB)
    start_idx = np.where(edc_dB <= -5)[0][0]
    end_idx = np.where(edc_dB <= threshold)[0][0]

    t = np.arange(len(edc_dB)) / fs
    t60 = multiplier * (t[end_idx] - t[start_idx])
    return t60


def reverberation_time(x, fs, freqs, plot=False) -> pd.DataFrame:
    results = []
    for frequency in freqs:
        bandpass = octave_bandpass(frequency, fs, fraction=3, order=2)
        filtered = sosfilt(bandpass, x)
        edc_dB = energy_decay_curve(filtered)
        edt = early_decay_time(edc_dB, fs)
        t30 = decay_time(edc_dB, fs)
        t20 = decay_time(edc_dB, fs, t20=True)
        c50 = clarity(filtered, fs, 50)
        c80 = clarity(filtered, fs, 80)
        results.append({
            'Frequency': frequency,
            'EDT': edt,
            'T20': t20,
            'T30': t30,
            'C50': c50,
            'C80': c80,
        })

        if plot:
            t = np.linspace(0, edc_dB.shape[-1]/fs, edc_dB.shape[-1])
            plt.plot(t, edc_dB, label=f'{frequency} Hz')

    if plot:
        plt.ylim(-80, 0)
        plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.legend(loc='upper right')
        plt.show()

    return pd.DataFrame.from_records(results)


def _plot_tolerances(rt60, freqs, fmin, fmax, ax: Axes):
    k4 = min(fmax, 4000)
    k8 = min(fmax, 8000)
    k20 = min(fmax, 20000)

    t30 = np.asarray(rt60)
    diff = np.insert(t30[:-1]-t30[1:], -1, 0.0)
    ymin, ymax = -0.05, +0.3
    ymin, ymax = np.min(diff), np.max(diff)

    ax.semilogx(freqs, diff, label='Measurement')

    ax.plot([63.0, 200.0], [0.3, 0.05], color='#555555')
    ax.hlines(
        [+0.05, -0.05, -0.1, -0.1, +0.3, +0.05, -0.05],
        [200, 100, k4,  k8, fmin, k8, fmin],
        [k8, k4, k8,  k20, 63, k20, 100],
        linestyles=['-', '-', '-', '--', '--', '--', '--'],
        colors='#555555',
        label='EBU Tech 3000'
    )

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title('Tolerance')
    ax.set_ylabel('Difference [s]')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim((fmin, fmax))
    ax.set_ylim((ymin-0.075, ymax+0.075))
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.margins(0, 0.1)
    ax.legend(loc='upper right')


def run(files, fmin, fmax, target=None):
    # ISO 1/3 octaves
    center_freqs = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
        20000
    ])

    center_freqs = center_freqs[(center_freqs >= fmin) & (center_freqs <= fmax)]

    file_times: list[pd.DataFrame] = []
    file_names: list[str] = []
    for path in files:
        file = pathlib.Path(path).absolute()
        fs, ir = wavread(file)
        rt60 = reverberation_time(ir, fs, center_freqs)

        file_times.append(rt60)
        file_names.append(file.stem[:4])

    num_plots = 1 if len(file_times) > 1 else 2
    _, axs = plt.subplots(num_plots, 1)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    # T60
    ax: Axes = axs if num_plots == 1 else axs[0]
    # ax.margins(0, 0.1)

    if len(file_times) > 1:
        for f, name in zip(file_times, file_names):
            ax.semilogx(center_freqs, f['T30'], label=f"T30: {name}")
    else:
        print(file_times[0].round(4).to_markdown(index=False))
        ax.semilogx(center_freqs, file_times[0]['EDT'], label='EDT')
        ax.semilogx(center_freqs, file_times[0]['T20'], label='T20')
        ax.semilogx(center_freqs, file_times[0]['T30'], label='T30')

    if target:
        ax.hlines(
            target,
            fmin,
            fmax,
            color='#555555',
            label=f"Target {target} s",
            linestyles='dashed',
        )

    ax.set_title('RT60')
    ax.set_ylabel('Decay [s]')
    ax.set_xlabel('Frequency [Hz]')
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim((fmin, fmax))
    ax.set_ylim((0, np.max(file_times[0]['T30'])+0.1))
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend(loc='upper right')

    if len(file_times) == 1:
        _plot_tolerances(file_times[0]['T30'], center_freqs, fmin, fmax, axs[1])

    plt.show()


@click.command(name='rt60', help='Plot RT60 decay times.')
@click.argument('filename', nargs=-1, type=click.Path(exists=True))
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--fmin', default=1.0)
@click.option('--fmax', default=1000.0)
@click.option('--target', default=0.0)
def main(filename, sim_dir, fmin, fmax, target):
    plt.rcParams.update(plot_styles)

    if sim_dir and len(filename) > 0:
        raise RuntimeError('--sim_dir not valid, when comparing IRs')

    files = filename
    if sim_dir:
        files = collect_wav_files(sim_dir, '*_out_normalised.wav')

    run(
        list(sorted(files)),
        fmin,
        fmax,
        target=target
    )
