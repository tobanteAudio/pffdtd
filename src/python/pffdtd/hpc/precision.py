# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
from decimal import Decimal
import numpy as np


@click.command(name='precision', help='Float precision')
def main():
    for bits in [8, 16, 24, 32, 64, 128, 256]:
        dynamic_range = (Decimal(2)**bits).log10()*20
        print(f'{bits:3d}-bit: {dynamic_range:.2f}dB')

    print()
    print('--------------- 32-bit ---------------')
    for clip_level in [0.00001, 0.05, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 128, 256]:
        clip_level = np.float32(clip_level)
        clip_dB = 20*np.log10(clip_level)
        nextafter = Decimal(float(np.nextafter(clip_level, 999.0)))
        step_dB = 20*(Decimal(1)+nextafter-Decimal(float(clip_level))).log10()
        print(f'{clip_dB:>+6.1f}dB = {step_dB:.12f}dB steps')

    print()
    print(f'16-bit step-size: {20.0*np.log10(1+(2.0/(2**16))):.10f}dB')
    print(f'24-bit step-size: {20.0*np.log10(1+(2.0/(2**24))):.10f}dB')
