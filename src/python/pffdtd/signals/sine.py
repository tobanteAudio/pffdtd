# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np

from pffdtd.common.wavfile import wavwrite
from pffdtd.signals.pink import crest_factor, normalize_to_RMS_dBFS


def generate_sine_wave(frequency, duration, fs, dB_rms=-20.0):
    n = int(fs*duration)
    t = np.linspace(0, duration, n, endpoint=False)
    x = np.sin(2 * np.pi * frequency * t)
    return normalize_to_RMS_dBFS(x, dB_rms)


@click.command(name='sine', help='Generate sine wave.')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--duration', default=10.0, type=float)
@click.option('--frequency', default=440.0, type=float)
@click.option('--fs', default=48000, type=int)
def main(output, duration, frequency, fs):
    x = generate_sine_wave(frequency, duration, fs)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))

    print(f'Peak:  {20*np.log10(peak):.1f} dBFS ({105+20*np.log10(peak):.1f} dBC SPL)')
    print(f'RMS:   {20*np.log10(rms):.1f} dBFS (85 dBC SPL)')
    print(f'Crest: {20*np.log10(crest_factor(x)):.2f} dB')
    print(f'Mean:  {abs(np.mean(x))}')

    wavwrite(output, fs, x)
