# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
from scipy.signal import butter, sosfilt, welch

from pffdtd.common.wavfile import wavwrite
from pffdtd.signals.level import crest_factor, normalize_to_RMS_dBFS


def generate_pink_noise(duration, fs, dbFS=-20.0, lowcut=20.0, highcut=20000.0, order=8):
    n = int(fs * duration)
    x = _generate_pink_noise_fft(n, fs)
    x = _bandlimit(x, fs, lowcut, highcut, order)
    x = normalize_to_RMS_dBFS(x, dbFS)
    return x


def pink_noise_slope(x, fs):
    # Estimate PSD
    f, Pxx = welch(x, fs, nperseg=4096)

    # Restrict to 20 Hz–20 kHz
    mask = (f >= 20) & (f <= 20000)
    logf = np.log10(f[mask])
    logP = np.log10(Pxx[mask])
    slope, _ = np.polyfit(logf, logP, 1)

    return slope


def _generate_pink_noise_fft(n, fs):
    freqs = np.fft.rfftfreq(n, 1/fs)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1/np.sqrt(freqs[1:])
    phases = np.exp(2j*np.pi*np.random.rand(len(freqs)))  # Random phase
    S = amplitude * phases
    x = np.fft.irfft(S, n=n)
    x = x / np.std(x)  # Normalize to unit-variance
    return x


def _bandlimit(x, fs, lowcut, highcut, order=8):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, x)


@click.command(name='pink', help='Generate pink noise')
@click.argument('output', nargs=1, type=click.Path())
@click.option('--duration', default=10.0, type=float)
@click.option('--fs', default=48000, type=int)
def main(output, duration, fs):
    x = generate_pink_noise(duration, fs)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))

    print(f'Peak:  {20*np.log10(peak):.1f} dBFS ({105+20*np.log10(peak):.1f} dBC SPL)')
    print(f'RMS:   {20*np.log10(rms):.1f} dBFS (85 dBC SPL)')
    print(f'Crest: {20*np.log10(crest_factor(x)):.2f} dB')
    print(f'Slope: {pink_noise_slope(x, fs)}')
    print(f'Mean:  {abs(np.mean(x))}')

    wavwrite(output, fs, x)
