# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


def main():
    cal = pd.read_csv(sys.argv[1], sep='\t', names=['Frequency', 'Offset'])

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


if __name__ == '__main__':
    main()
