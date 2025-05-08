# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.signal import butter, firwin2, unit_impulse, sosfilt, windows

from pffdtd.signals.iir import dolby_atmos_target_curve, linkwitz_riley_crossover
from pffdtd.signals.group_delay import group_delay_seconds


def linear_phase_from_sos(sos, ntaps, window='hann'):
    assert ntaps % 2 != 0

    taps = np.zeros(ntaps)
    taps[ntaps//2] = 1.0

    taps = sosfilt(sos, x=taps)
    taps = sosfilt(sos, x=taps[::-1])[::-1]
    taps *= windows.get_window(window, taps.shape[-1], fftbins=False)

    nfft = ntaps
    H = np.fft.rfft(taps, nfft)
    H = np.sqrt(np.abs(H))*np.exp(1j*np.angle(H))
    return np.fft.irfft(H, nfft)


@click.command(name='fir', help='FIR filters.')
@click.option('--fs', default=48000.0, type=float)
@click.option('--ntaps', default=1001, type=int)
def main(fs, ntaps):
    group_delay = ntaps//2
    group_delay_ms = 1000/fs*group_delay
    resolution = fs/ntaps
    print(f'{ntaps=}')
    print(f'{group_delay_ms=:.3f} ms')
    print(f'{resolution=:.2f} Hz')

    nfft = int(round(fs*16))
    freqs = np.fft.rfftfreq(nfft, 1/fs)

    sos, _ = linkwitz_riley_crossover(80, fs, order=4)
    sos = np.concatenate([
        # butter(4, 8, 'highpass', fs=fs, output='sos'),
        dolby_atmos_target_curve(fs),
    ])

    sos_h = np.fft.rfft(sosfilt(sos, unit_impulse(nfft)))
    sos_mag = np.maximum(np.abs(sos_h), 10**(-144/20))
    sos_dB = 20*np.log10(sos_mag)

    lin = linear_phase_from_sos(sos, ntaps)
    lin_h = np.fft.rfft(lin, nfft)
    lin_dB = 20*np.log10(np.maximum(np.abs(lin_h), 1e-9))

    firw = firwin2(ntaps, freqs, sos_mag, fs=fs)
    firw_h = np.fft.rfft(firw, nfft)
    firw_dB = 20*np.log10(np.maximum(np.abs(firw_h), 1e-9))

    _, axs = plt.subplots(4, 1)

    ax: Axes = axs[0]
    ax.set_title('Magnitude Response')
    ax.semilogx(freqs, firw_dB, label='FIRWIN2')
    ax.semilogx(freqs, lin_dB, label='SOS-to-LIN')
    ax.semilogx(freqs, sos_dB, label='SOS')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlim(1, 20_000)
    ax.set_ylim(-100, 5)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    ax: Axes = axs[1]
    ax.set_title('Error')
    ax.semilogx(freqs, firw_dB-sos_dB, label='FIRW-SOS')
    ax.semilogx(freqs, lin_dB-sos_dB, label='LIN-SOS')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlim(1, 20_000)
    ax.set_ylim(-10, 10)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    ax: Axes = axs[2]
    ax.set_title('|Error|')
    ax.semilogx(freqs, np.abs(firw_dB-sos_dB), label='FIRW-SOS')
    ax.semilogx(freqs, np.abs(lin_dB-sos_dB), label='LIN-SOS')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlim(1, 20_000)
    ax.set_ylim(0, 10)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    ax: Axes = axs[3]
    ax.set_title('Group Delay')
    ax.semilogx(freqs, group_delay_seconds(sos_h, freqs)*1000, label='SOS')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Group Delay [ms]')
    ax.set_xlim(1, 20_000)
    ax.set_ylim(-10, 100)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    plt.show()

    t = np.arange(ntaps)/fs*1000-group_delay_ms
    _, axs = plt.subplots(2, 1)
    ax: Axes = axs[0]
    ax.set_title('Impulse Response')
    ax.plot(t, firw, label='FIRWIN2')
    ax.plot(t, lin, label='LIN-to-SOS')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Magnitude [FS]')
    # ax.set_xlim(1, 20_000)
    # ax.set_ylim(-145, 10)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    ax: Axes = axs[1]
    ax.set_title('Impulse Response')
    ax.plot(t, 20*np.log10(np.maximum(np.abs(firw), 1e-9)), label='FIRWIN2')
    ax.plot(t, 20*np.log10(np.maximum(np.abs(lin), 1e-9)), label='LIN-to-SOS')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Magnitude [dB]')
    # ax.set_xlim(1, 20_000)
    ax.set_ylim(-145, 10)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()
    plt.show()
