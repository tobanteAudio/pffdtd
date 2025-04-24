# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks

from pffdtd.common.plot import plot_styles


def main():
    c = 343.0
    delta_L = 0.83
    tau = delta_L / c

    A_d = 1.0
    A_r = 10**(-1.86/20)*(1-0.6)
    A_r = 10**(-1.86/20)
    A_r = 10**(0/20)

    # Frequency range for analysis (e.g., 0 Hz to 5000 Hz)
    frequencies = np.linspace(20, 2_000, 10000)

    # Calculate the complex representations
    direct_signal = A_d * np.ones_like(frequencies)
    reflected_signal = A_r * np.exp(-1j * 2 * np.pi * frequencies * tau)
    sum_signal = direct_signal + reflected_signal

    amplitude = np.abs(sum_signal)
    phase = np.angle(sum_signal)
    phase_deg = np.rad2deg(phase)

    amplitude_dB = 20*np.log10(amplitude)

    minima, _ = find_peaks(-amplitude_dB, height=0)
    print(frequencies[minima])

    plt.rcParams.update(plot_styles)
    _, axs = plt.subplots(2, 1, sharex='all')
    ax0: Axes = axs[0]
    ax1: Axes = axs[1]

    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    ax0.semilogx(frequencies, 20*np.log10(np.abs(direct_signal)), label='Direct')
    ax0.semilogx(frequencies, 20*np.log10(np.abs(reflected_signal)), label='Reflected')
    ax0.semilogx(frequencies, amplitude_dB, label='Sum')
    ax0.vlines(frequencies[minima], np.min(amplitude_dB), np.max(amplitude_dB), linestyles='--')
    ax0.set_xlim(frequencies[0], frequencies[-1])
    # ax0.set_ylim(-40, 10)
    ax0.set_ylabel('Amplitude [dB]')
    ax0.set_title('Amplitude Response')
    ax0.xaxis.set_major_formatter(formatter)
    ax0.grid(which='major', linewidth=0.75)
    ax0.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax0.minorticks_on()
    ax0.legend()

    ax1.semilogx(frequencies, phase_deg)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Phase [degrees]')
    ax1.set_title('Phase Response')
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(which='major', linewidth=0.75)
    ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax1.minorticks_on()

    plt.show()


if __name__ == '__main__':
    main()
