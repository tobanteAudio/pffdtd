# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
from scipy.signal import butter, sosfilt

from pffdtd.common.wavfile import wavwrite


def generate_pink_noise(duration, fs, dbFS=-20.0, lowcut=20.0, highcut=20000.0):
    n = int(fs * duration)
    x = _generate_pink_noise_fft(n, fs)
    x = _bandlimit(x, fs, lowcut, highcut)
    x = _normalize_to_dBFS(x, dbFS)
    return x


def _generate_pink_noise_fft(n, fs):
    freqs = np.fft.rfftfreq(n, 1/fs)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1/np.sqrt(freqs[1:])
    phases = np.exp(2j*np.pi*np.random.rand(len(freqs)))  # Random phase
    S = amplitude * phases
    x = np.fft.irfft(S, n=n)
    x = x / np.std(x)  # Normalize to unit-variance
    return x


def _bandlimit(x, fs, lowcut, highcut, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, x)


def _normalize_to_dBFS(x, target_dBFS):
    rms = np.sqrt(np.mean(x**2))
    current_dB = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dB
    gain = 10**(gain_dB / 20)
    return x * gain


@click.command(name='pink', help='Generate pink noise')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--duration', default=10.0, type=float)
@click.option('--fs', default=48000, type=int)
def main(output, duration, fs):
    x = generate_pink_noise(duration, fs)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    print(f'Peak:  {20*np.log10(peak):.1f} dB')
    print(f'RMS:   {20*np.log10(rms):.1f} dB')
    print(f'Crest: {peak/rms:.2f}')

    wavwrite(output, fs, x)
