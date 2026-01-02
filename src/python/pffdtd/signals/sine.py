# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np

from pffdtd.signals.pink import crest_factor, normalize_to_RMS_dBFS
from pffdtd.signals.wavfile import wavwrite


def generate_sine_wave(frequency, duration, fs, dB_rms=-20.0):
    n = int(fs*duration)
    t = np.linspace(0, duration, n, endpoint=False)
    x = np.sin(2 * np.pi * frequency * t)
    return normalize_to_RMS_dBFS(x, dB_rms)


def generate_exponential_sine_sweep(T, f1, f2, fs, fade_ms=50) -> tuple[np.ndarray, np.ndarray]:
    """Generate Farina-style exponential (logarithmic) sine sweep and its inverse.

    - https://www.melaudia.net/zdoc/sweepSine.PDF
    - https://dsp.stackexchange.com/questions/41696/calculating-the-inverse-filter-for-the-exponential-sine-sweep-method

    Parameters
    ----------
    T : duration in seconds
    f0 : start frequency in hertz
    f1 : end frequency in hertz
    fs : sample-rate

    Returns
    -------
    sweep : excitation signal
    inv : inverse filter (for convolution with the recorded signal)
    """
    R = np.log(f2 / f1)
    t = np.arange(0, int(T * fs)) / fs

    phase = 2 * np.pi * f1 * T / R * (np.exp(R * t / T) - 1.0)
    sweep = np.sin(phase)
    sweep /= np.max(np.abs(sweep))

    fade_len = int(fade_ms / 1000.0 * fs)
    if fade_len > 0:
        fade = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        window = np.ones_like(sweep)
        window[:fade_len] = fade
        window[-fade_len:] = fade[::-1]
        sweep *= window

    k = np.exp(R * t / T)
    inv = sweep[::-1] / k
    inv /= np.max(np.abs(inv))

    return sweep, inv


@click.command(name='sine', help='Generate sine wave.')
@click.argument('output', nargs=-1, type=click.Path())
@click.option('--duration', default=10.0, type=float)
@click.option('--frequency', default=440.0, type=float)
@click.option('--fs', default=48000, type=int)
@click.option('--rms_dbFS', 'rms_dbFS', default=-20.0, type=float)
def main(output, duration, frequency, fs, rms_dbFS):
    x = generate_sine_wave(frequency, duration, fs, rms_dbFS)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))

    print(f'Peak:  {20*np.log10(peak):.1f} dBFS ({105+20*np.log10(peak):.1f} dBC SPL)')
    print(f'RMS:   {20*np.log10(rms):.1f} dBFS (85 dBC SPL)')
    print(f'Crest: {20*np.log10(crest_factor(x)):.2f} dB')
    print(f'Mean:  {abs(np.mean(x)):.6f}')

    if len(output) == 1:
        wavwrite(output[0], fs, x)
