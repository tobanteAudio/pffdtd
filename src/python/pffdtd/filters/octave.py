# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
from scipy import signal


def octave_bandpass(center: float, fs: float, fraction: float = 3, order: int = 2) -> np.ndarray:
    """One-third octave by default.
    """
    factor = 2 ** (1/(fraction*2))
    low = center / factor
    high = center * factor
    return signal.butter(order, [low, high], btype='band', fs=fs, output='sos')


def octave_smoothing(magnitudes, fs, nfft, fraction=3):
    """
    Apply fractional octave smoothing to FFT magnitudes.

    Parameters:
        - magnitudes: Array of FFT magnitudes.
        - fs: Sampling rate of the signal.
        - nfft: Size of the FFT.
        - fraction: Fraction of the octave for smoothing (e.g., 3 for 1/3 octave, 6 for 1/6 octave).

    Returns:
        - smoothed: Array of smoothed FFT magnitudes.
    """
    # smoothed = np.zeros_like(magnitudes)

    # for i in tqdm(range(magnitudes.shape[-1])):
    #     fc = frequencies[i]
    #     fl = fc / 2**(1/(2*fraction))
    #     fu = fc * 2**(1/(2*fraction))
    #     indices = np.where((frequencies >= fl) & (frequencies <= fu))[0]
    #     if len(indices) > 0:
    #         smoothed[i] = np.mean(magnitudes[indices])

    # return smoothed

    # 1) Compute center freqs and their lower/upper band edges
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    factor = 2**(1/(2*fraction))
    fl = freqs / factor
    fu = freqs * factor

    # 2) Find, for each center freq, the index of the first bin ≥ fl and
    #    the last bin ≤ fu using binary search (O(N log N))
    idx_l = np.searchsorted(freqs, fl, side='left')
    idx_u = np.searchsorted(freqs, fu, side='right') - 1

    # 3) Clip any reversed windows (where fu < fl) to empty
    valid = idx_u >= idx_l
    widths = idx_u - idx_l + 1  # only meaningful when valid is True

    # 4) Build a prefix‐sum of the magnitude array (O(N))
    #    pad with a zero at the front so that
    #      sum over [a..b] = cumsum[b+1] - cumsum[a]
    cumsum = np.concatenate(([0.], np.cumsum(magnitudes, dtype=float)))

    # 5) Allocate output and fill only the valid bins (O(N))
    smoothed = np.zeros_like(magnitudes)
    # vectorized grab of sums at upper & lower edges:
    sum_u = cumsum[idx_u + 1]
    sum_l = cumsum[idx_l]
    smoothed[valid] = (sum_u[valid] - sum_l[valid]) / widths[valid]

    return smoothed


def center_frequencies(divisions=12, f_ref=1000, oct_down=6, oct_up=5):
    """Returns the center frequencies with `divisions` bands per octave.
    """
    exp = np.arange(-oct_down, oct_up, step=1.0/divisions)
    freqs = f_ref * (2.0 ** exp)
    return freqs


@click.command(name='octave', help='Octave utilities.')
def main():
    freqs = center_frequencies(12, f_ref=440, oct_up=6)
    freqs = freqs[(freqs >= 20.0) & (freqs <= 20_000.0)]
    print(f"total bands: {len(freqs)}")
    print('first 5:', np.round(freqs[:5], 2))
    print('around 1 kHz:', freqs[np.abs(freqs-1000).argmin()])
    print('last 5:', np.round(freqs[-5:], 2))
