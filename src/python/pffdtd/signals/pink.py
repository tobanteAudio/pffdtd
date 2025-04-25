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
    amplitude = np.where(freqs == 0, 0, 1/np.sqrt(freqs+0.000001))  # Avoid divide-by-zero at DC
    phases = np.exp(2j*np.pi*np.random.rand(len(freqs)))  # Random phase

    S = amplitude * phases
    x = np.fft.irfft(S, n=n)
    x = x / np.std(x)  # Normalize to unit-variance
    return x


def _bandlimit(x, fs, lowcut, highcut, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, x)  # for zero-phase use filtfilt if you prefer


def _normalize_to_dBFS(x, target_dBFS):
    rms = np.sqrt(np.mean(x**2))
    current_dB = 20 * np.log10(rms)
    gain_dB = target_dBFS - current_dB
    gain = 10**(gain_dB / 20)
    return x * gain


@click.command(name='pink', help='Generate pink noise')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--duration', default=60.0, type=float)
@click.option('--fs', default=48000, type=int)
def main(output, duration, fs):
    pink = generate_pink_noise(duration, fs)
    print(np.max(pink))
    print(np.min(pink))
    wavwrite(output, fs, pink)
