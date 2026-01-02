# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# --- 1. Load impulse response from WAV ---
fs, ir = wavfile.read(sys.argv[1])

# If stereo, take left channel
if ir.ndim > 1:
    ir = ir[:, 0]

# Convert to float and normalize to avoid overflow
ir = ir.astype(np.float64)
ir /= np.max(np.abs(ir))

# Time axis
t = np.arange(len(ir)) / fs

# --- 2. Variant A: Simple energy-based ETC ---
energy = ir**2
energy /= np.max(energy)  # normalize
# Add tiny epsilon to avoid log(0)
eps = 1e-20
etc_energy_db = 10 * np.log10(energy + eps)

# --- 3. Variant B: Envelope-based ETC using Hilbert transform ---
analytic = hilbert(ir)
envelope = np.abs(analytic)
envelope /= np.max(envelope)  # normalize
etc_envelope_db = 20 * np.log10(envelope + eps)


def moving_average(x, N):
    # Simple causal moving average of length N
    return np.convolve(x, np.ones(N)/N, mode='same')


# Example: smooth over 0.1 ms window
window_ms = 0.1
N = int(fs * window_ms / 1000.0)
etc_smoothed = moving_average(etc_envelope_db, max(N, 1))
t0 = np.argmax(etc_smoothed)/fs
print(t0)

# --- 4. Plot results ---
plt.figure()
# plt.plot(t, etc_energy_db, label="ETC (energy, 10*log10)")
plt.plot((t-t0)*1000, etc_smoothed, label='ETC (envelope, 20*log10)')
plt.xlabel('Time [ms]')
plt.ylabel('Level [dB]')
plt.title('Energy Time Curve (ETC) from Impulse Response')
plt.xlim(-80, 280)
plt.ylim(-180, 10)
plt.legend()
plt.grid(True)
plt.show()
