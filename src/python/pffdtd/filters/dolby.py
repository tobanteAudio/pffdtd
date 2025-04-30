# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import numpy as np
import matplotlib.pyplot as plt


from pffdtd.filters.fir import linear_phase_from_sos
from pffdtd.filters.iir import peak_filter


def dolby_atmos_target_curve(fs):
    """Returns an SOS filter matching the Dolby Atmos Target Curve with a maximum error of +/-0.5dB

    - https://gracedesign.com/support/manuals/m908_Atmos_Target_Curve_EQ.pdf
    """
    return np.concatenate([
        peak_filter(16, 10**(-2.3/20), 1.52, fs),
        peak_filter(71, 10**(+1.2/20), 0.44, fs),
        peak_filter(185, 10**(-0.6/20), 0.86, fs),
        peak_filter(5770, 10**(-1.4/20), 0.6, fs),
        peak_filter(20000, 10**(-6.4/20), 0.2, fs),
    ])


@click.command(name='dolby', help='Dolby Atmos Target Curve.')
def main():
    fs = 48000
    ntaps = 501
    group_delay = ntaps//2
    sos = dolby_atmos_target_curve(fs)
    taps = linear_phase_from_sos(sos, ntaps)

    nfft = max(ntaps, 2048)
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    H = np.fft.rfft(taps, nfft)
    dB = 20*np.log10(np.abs(H)+np.finfo(np.float64).eps)

    plt.semilogx(freqs, dB, label=f'FIR: {ntaps}')
    plt.xlim(10, 22_000)
    plt.ylim(-10, 10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Dolby Target Curve Response @ {fs/ntaps:.2f} Hz Resolution')
    plt.grid(which='both')
    plt.legend()
    plt.show()

    plt.plot(np.arange(ntaps), taps[:ntaps])
    plt.title(f'Dolby Target Curve Impulse @ {1000/fs*group_delay:.2f} ms Latency')
    plt.grid(which='both')
    plt.show()
