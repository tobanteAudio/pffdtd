# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks


def minimum_phase_reconstruction(M_half: np.ndarray) -> np.ndarray:
    """
    Reconstructs a minimum phase impulse response with a magnitude
    response matching M

    Parameters:
        M_half: Of length (N//2 + 1), containing magnitudes at frequencies 0, 2π/N, 4π/N, …, π (Nyquist).

    Returns:
        out: Minimum phase impulse response
    """
    # infer full FFT length N (must be even)
    N = (len(M_half) - 1) * 2

    # 1. Reconstruct full, even‐symmetric magnitude spectrum
    #   bins 0 … N/2
    #   then bins N/2−1 … 1
    M_full = np.concatenate([
        M_half,
        M_half[-2:0:-1]   # skip the last (Nyquist) and the first (DC)
    ])

    # 2. Log‐magnitude and real cepstrum
    eps = 1e-12                                 # avoid log(0)
    L = np.log(np.maximum(M_full, eps))         # length-N real, symmetric
    c = np.fft.ifft(L).real                     # real cepstrum, length N

    # 3. Build the minimum‐phase cepstrum
    c_min = np.zeros_like(c)
    c_min[0] = c[0]                             # keep the DC term
    # double the causal part 1 … N/2−1
    c_min[1:N//2] = 2 * c[1:N//2]
    # if you want to preserve the Nyquist term (for even N), uncomment:
    c_min[N//2] = c[N//2]

    # 4. Re‐synthesize the complex spectrum
    #    H_min[k] = exp( FFT{c_min} )
    H_min = np.exp(np.fft.fft(c_min))
    out = np.fft.ifft(H_min).real
    return out


@click.command(name='phase', help='Plot phase mismatch.')
def main() -> None:
    c = 343.0
    frequencies = np.linspace(20, 2_000, 10000)

    def _reflect(delta_m, gain):
        tau = delta_m/c
        return gain * np.exp(-1j * 2 * np.pi * frequencies * tau)

    direct = 1.0 * np.ones_like(frequencies)
    reflected_A = _reflect(1.8, 10**(-12/20))
    reflected_B = _reflect(0.51, 10**(-6/20))
    mix = direct + reflected_A + reflected_B

    magnitude = np.abs(mix)
    magnitude_dB = 20*np.log10(magnitude)

    minima, _ = find_peaks(-magnitude_dB, height=0)
    print(frequencies[minima])

    _, axs = plt.subplots(2, 1, sharex='all')
    ax0: Axes = axs[0]
    ax1: Axes = axs[1]

    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    ax0.semilogx(frequencies, 20*np.log10(np.abs(direct)), label='Direct')
    ax0.semilogx(frequencies, 20*np.log10(np.abs(reflected_A)), label='Floor')
    ax0.semilogx(frequencies, magnitude_dB, label='Mix')
    ax0.vlines(frequencies[minima], np.min(magnitude_dB), np.max(magnitude_dB), linestyles='--')
    ax0.set_xlim(frequencies[0], frequencies[-1])
    ax0.set_ylim(-40, 10)
    ax0.set_ylabel('Magnitude [dB]')
    ax0.set_title('Magnitude')
    ax0.xaxis.set_major_formatter(formatter)
    ax0.grid(which='major', linewidth=0.75)
    ax0.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax0.minorticks_on()
    ax0.legend()

    ax1.semilogx(frequencies, np.rad2deg(np.angle(mix)))
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Phase [degrees]')
    ax1.set_title('Phase')
    ax1.set_ylim(-180, 180)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(which='major', linewidth=0.75)
    ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    plt.show()
