# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
import numpy as np
from scipy.signal import oaconvolve

from pffdtd.common.wavfile import wavread, wavwrite
from pffdtd.signals.level import normalize_to_RMS_dBFS
from pffdtd.signals.pink import generate_pink_noise


@click.command(name='rew', help='Export IR as REW offline measurement.')
@click.argument('impulse_path', nargs=1, type=click.Path(exists=True))
@click.option('--stimulus', type=click.Path(exists=True))
def main(impulse_path, stimulus):
    fs_ir, ir = wavread(impulse_path)
    ir = ir / np.sqrt(np.sum(ir**2))

    fs_sweep, sweep = wavread(stimulus)
    assert fs_ir == fs_sweep

    def _summary(x, name):
        peak = np.max(np.abs(x))
        rms = np.sqrt(np.mean(x**2))

        print(f'{name}:')
        print(f'  - Samples:  {len(x)}')
        print(f'  - Duration: {len(x)/fs_ir:.3f} s')
        print(f'  - Peak:     {20*np.log10(peak):.2f} dB')
        print(f'  - RMS:      {20*np.log10(rms):.2f} dB')

    pink = generate_pink_noise(10.0, fs_ir)
    level_ref = oaconvolve(pink, ir, mode='full')
    measurement = oaconvolve(sweep, ir, mode='full')
    measurement = normalize_to_RMS_dBFS(measurement, -32)
    measurement = (measurement/np.max(np.abs(measurement)))*(10**((85-120)/20))

    _summary(ir, 'IR')
    _summary(sweep, 'Sweep')
    _summary(pink, 'Pink')
    _summary(level_ref, 'Level Ref')
    _summary(measurement, 'Measurement')

    wavwrite('out.wav', fs_ir, measurement)

    # plot_impulse_response_summary(ir, fs, fmax=fmax)
    # plt.show()
