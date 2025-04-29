# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import sosfilt

from pffdtd.filters.iir import linkwitz_riley_crossover
from pffdtd.filters.phase import minimum_phase_reconstruction

sb_tw29dn_b_8_94db = [
    (0.0, 0.001),
    (20.0, 0.001),
    (2000.0, 0.6),
    (2100.0, 0.7),
    (2200.0, 0.55),
    (2300.0, 0.65),
    (2500.0, 0.55),
    (2700.0, 0.5),
    (3000.0, 0.45),
    (4000.0, 0.38),
    (4500.0, 0.29),
    (4750.0, 0.42),
    (5000.0, 0.28),
    (5100.0, 0.42),
    (5150.0, 0.40),
    (5500.0, 0.35),
    (6000.0, 0.3),
    (7000.0, 0.15),
    (7500.0, 0.05),
    (8500.0, 0.12),
    (9000.0, 0.05),
    (10000.0, 0.6),
]

morel_1044_94db = [
    (0.0, 0.001),
    (20.0, 0.001),
    (975, 7.5),
    (1000, 7.5),
    (1005, 7.5),
    (1100, 0.4),
    (1200, 0.45),
    (1300, 0.45),
    (1500, 0.4),
    (1800, 0.4),
    (2000, 0.2),
    (2100, 0.3),
    (2250, 0.4),
    (3000, 0.4),
    (3750, 0.5),
    (3800, 0.6),
    (4000, 0.4),
    (4300, 0.5),
    (4750, 0.4),
    (5000, 0.35),
    (5800, 0.1),
    (6200, 0.2),
    (7000, 0.4),
    (7900, 0.5),
    (8100, 0.4),
    (8200, 0.5),
    (9000, 0.6),
    (10000, 0.7),
]

volt_vm_572_94db = [
    (0.0, 0.001),
    (20.0, 0.001),
    (500.0, 1.2),
    (600.0, 1.8),
    (700.0, 0.7),
    (800.0, 0.45),
    (900.0, 0.35),
    (950.0, 0.4),
    (1000.0, 0.35),
    (1250.0, 0.5),
    (1750.0, 0.25),
    (2000.0, 0.3),
    (2400.0, 0.2),
    (2500.0, 0.35),
    (2750.0, 0.15),
    (3000.0, 0.35),
    (3300.0, 0.175),
    (3750.0, 0.5),
    (4000.0, 0.15),
    (4200.0, 0.25),
    (4500.0, 0.15),
    (4750.0, 0.6),
    (5000.0, 0.25),
    (5500.0, 0.075),
    (6000.0, 0.150),
    (6250.0, 0.30),
    (6750.0, 0.5),
    (6800.0, 0.250),
    (6900.0, 0.150),
    (7500.0, 0.35),
    (8500.0, 0.325),
    (10000.0, 0.3),
]

volt_vm_752_94db = [
    (500, 0.5),
    (540, 0.7),
    (560, 0.6),
    (600, 0.5),
    (620, 0.4),
    (640, 0.55),
    (700, 0.4),
    (800, 0.4),
    (900, 0.45),
    (1000, 0.5),
    (1500, 0.5),
    (1750, 0.2),
    (1900, 0.6),
    (2000, 0.6),
    (2250, 0.3),
    (2500, 0.3),
    (2750, 0.2),
    (3000, 0.3),
    (3100, 0.2),
    (3200, 0.3),
    (4000, 0.1),
    (4200, 0.2),
    (4500, 0.1),
    (4900, 1.2),
    (5000, 1.2),
    (5100, 0.4),
    (5200, 0.3),
    (5500, 0.35),
    (6000, 0.2),
    (6500, 0.4),
    (7000, 0.1),
    (8000, 0.3),
    (9000, 0.1),
    (10000, 0.1),
]

scan_speak_32w_4878t00_94db = [
    (0.0, 0.001),
    (20.0, 0.001),
    (50, 6*(1/20)),
    (55, 4*(1/20)),
    (60, 4*(1/20)),
    (63, 6*(1/20)),
    (67, 2*(1/20)),
    (70, 2*(1/20)),
    (75, 3*(1/20)),
    (80, 2*(1/20)),
    (90, 2.5*(1/20)),
    (100, 2*(1/20)),
    (125, 3*(1/20)),
    (150, 2*(1/20)),
    (180, 4*(1/20)),
    (210, 3*(1/20)),
    (225, 4.5*(1/20)),
    (240, 3*(1/20)),
    (275, 3*(1/20)),
    (300, 4*(1/20)),
    (400, 6*(1/20)),
    (450, 11*(1/20)),
    (475, 6*(1/20)),
    (500, 10*(1/20)),
    (600, 8*(1/20)),
    (700, 9*(1/20)),
    (750, 3*(1/20)),
    (800, 3*(1/20)),
    (850, 7*(1/20)),
    (900, 7*(1/20)),
    (950, 12*(1/20)),
    (1000, 15*(1/20)),
    (1100, 25*(1/20)),
    (1200, 5*(1/20)),
    (1300, 10*(1/20)),
    (1400, 5*(1/20)),
    (1500, 7*(1/20)),
    (1600, 4*(1/20)),
    (1900, 2*(1/20)),
    (2000, 1*(1/20)),
    (10000, 1*(1/20)),
]

radian_950pb_104db = [
    (0.0, 0.001),
    (20.0, 0.001),
    (800, 0.7),
    (850, 0.8),
    (900, 0.75),
    (950, 0.9),
    (1000, 0.95),
    (1500, 1.4),
    (1750, 1.5),
    (2000, 1.3),
    (2500, 1.8),
    (2750, 1.3),
    (3000, 1.3),
    (3100, 1.5),
    (3300, 0.9),
    (3800, 1.25),
    (4000, 0.95),
    (4200, 1.1),
    (4300, 0.7),
    (4500, 1.0),
    (4750, 1.0),
    (5000, 0.7),
    (5250, 1.1),
    (5750, 0.9),
    (6000, 0.85),
    (7000, 0.5),
    (7500, 1.1),
    (8000, 1.6),
    (9000, 0.9),
    (9500, 2.5),
    (10000, 1.7),
]

bc_14na100_104db = [
    (0.0, 0.001),
    (19, 4.25),
    (20, 4.25),
    (25, 3.25),
    (30, 2.5),
    (40, 1.5),
    (50, 1.0),
    (60, 0.85),
    (70, 0.8),
    (80, 0.78),
    (90, 0.85),
    (100, 1.0),
    (125, 1.25),
    (175, 1.15),
    (200, 1.25),
    (250, 1.3),
    (300, 1.1),
    (325, 0.7),
    (375, 1.0),
    (390, 0.8),
    (400, 1.0),
    (450, 0.8),
    (475, 1.1),
    (500, 0.8),
    (525, 1.2),
    (550, 0.8),
    (590, 1.1),
    (600, 1.0),
    (700, 1.2),
    (800, 0.9),
    (900, 1.2),
    (950, 0.55),
    (1000, 1.1),
    (1100, 1.0),
    (1250, 0.6),
    (1500, 0.75),
    (2000, 0.25),
    (2250, 1.1),
    (2500, 0.3),
    (2750, 0.7),
    (3000, 0.2),
    (4000, 0.2),
    (5000, 0.002),
]


@click.command(name='distortion', help='Driver distortion')
def main():
    fs = 48000
    fftfreqs = np.fft.rfftfreq(fs, 1/fs)

    def interpolate(freqs, dist):
        return np.maximum(0.0, interp1d(
            freqs,
            dist,
            kind='cubic',
            bounds_error=False,
            fill_value=(dist[0], dist[-1]),
        )(fftfreqs))

    def db_to_percent(db):
        return 100 * 10 ** (db / 20)

    def percent_to_db(pct):
        return 20 * np.log10(pct+0.000001 / 100)

    woofer = interpolate([f for f, _ in scan_speak_32w_4878t00_94db], [d for _, d in scan_speak_32w_4878t00_94db])
    midrange = interpolate([f for f, _ in volt_vm_752_94db], [d for _, d in volt_vm_752_94db])
    tweeter = interpolate([f for f, _ in morel_1044_94db], [d for _, d in morel_1044_94db])

    ax = plt.gca()
    ax.semilogx(fftfreqs, 20*np.log10(woofer/100+np.finfo(np.float64).eps), label='Woofer')
    ax.semilogx(fftfreqs, 20*np.log10(midrange/100+np.finfo(np.float64).eps), label='Midrange')
    ax.semilogx(fftfreqs, 20*np.log10(tweeter/100+np.finfo(np.float64).eps), label='Tweeter')
    ax.set_xlim(20.0, 10000.0)
    ax.set_ylim(-80.0, 0.0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title('2nd Harmonic')
    secax = ax.secondary_yaxis('right', functions=(db_to_percent, percent_to_db))
    secax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    secax.set_ylabel('Magnitude [%]')
    ax.grid(which='both')
    ax.legend()
    plt.tight_layout()
    plt.show()

    woofer_min = minimum_phase_reconstruction(woofer/100)
    midrange_min = minimum_phase_reconstruction(midrange/100)
    tweeter_min = minimum_phase_reconstruction(tweeter/100)

    lowpass_b1, highpass_b1 = linkwitz_riley_crossover(800, fs, 4)
    lowpass_b2, highpass_b2 = linkwitz_riley_crossover(3800, fs, 4)

    woofer_filt = sosfilt(lowpass_b1, woofer_min)
    midrange_filt = sosfilt(highpass_b1, sosfilt(lowpass_b2, midrange_min))
    tweeter_filt = sosfilt(highpass_b2, tweeter_min)

    mix = woofer_filt+midrange_filt+tweeter_filt

    H_woofer = np.fft.rfft(woofer_filt, fs)
    H_midrange = np.fft.rfft(midrange_filt, fs)
    H_tweeter = np.fft.rfft(tweeter_filt, fs)
    H_mix = np.fft.rfft(mix, fs)

    ax = plt.gca()
    ax.semilogx(fftfreqs, 20*np.log10(np.abs(H_woofer)), linestyle='--', label='Woofer')
    ax.semilogx(fftfreqs, 20*np.log10(np.abs(H_midrange)), linestyle='--', label='Midrange')
    ax.semilogx(fftfreqs, 20*np.log10(np.abs(H_tweeter)), linestyle='--', label='Tweeter')
    ax.semilogx(fftfreqs, 20*np.log10(np.abs(H_mix)), label='Mix')
    ax.set_xlim(20.0, 10000.0)
    ax.set_ylim(-80.0, 0.0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title('2nd Harmonic')
    secax = ax.secondary_yaxis('right', functions=(db_to_percent, percent_to_db))
    secax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    secax.set_ylabel('Magnitude [%]')
    ax.grid(which='both')
    ax.legend()
    plt.tight_layout()
    plt.show()
