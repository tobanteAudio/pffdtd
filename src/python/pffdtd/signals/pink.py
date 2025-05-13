# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
from scipy.signal import butter, sosfilt, welch

from pffdtd.signals.level import crest_factor, normalize_to_RMS_dBFS
from pffdtd.signals.wavfile import wavwrite


def generate_pink_noise(
    duration: float,
    fs: float,
    *,
    rms_dB: float = -20.0,
    lowcut: float = 20.0,
    highcut: float = 20000.0,
    order: int = 8,
    rng: np.random.Generator | None = None
):
    if not rng:
        rng = np.random.default_rng()

    # Synthesize
    n = int(fs * duration)
    freqs = np.fft.rfftfreq(n, 1/fs)
    amplitude = np.zeros_like(freqs)
    amplitude[1:] = 1/np.sqrt(freqs[1:])
    phases = np.exp(2j*np.pi*rng.random(len(freqs)))
    S = amplitude * phases
    x = np.fft.irfft(S, n=n)
    x = x / np.std(x)  # Normalize to unit-variance

    # Bandlimit
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    x = sosfilt(sos, x)

    # Normalize
    x = normalize_to_RMS_dBFS(x, rms_dB)

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


@click.command(name='pink', help='Generate pink noise')
@click.argument('output', nargs=-1, type=click.Path())
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

    if len(output) == 1:
        wavwrite(output[0], fs, x)
