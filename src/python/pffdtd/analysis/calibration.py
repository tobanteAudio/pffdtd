# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


@click.command(name='calibration', help='Load microphone calibration file.')
@click.argument('calibration_csv', nargs=1, type=click.Path(exists=True))
def main(calibration_csv):
    cal = pd.read_csv(calibration_csv, sep='\t', names=['Frequency', 'Offset'])

    fs = 48000
    nfft = 4096*4
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    resampled = interp1d(
        cal['Frequency'],
        cal['Offset'],
        kind='cubic',
        bounds_error=False,
        fill_value=(cal['Offset'].iloc[0], cal['Offset'].iloc[-1]),
    )(freqs)

    print(cal)
    print(pd.DataFrame.from_dict({'Frequency': freqs, 'Offset': resampled}))

    plt.semilogx(cal['Frequency'], cal['Offset'], label='Original')
    plt.semilogx(freqs, resampled, label='Resampled')
    plt.grid(which='both')
    plt.show()
