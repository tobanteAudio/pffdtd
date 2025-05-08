# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.signals import convolution
from pffdtd.signals import fir
from pffdtd.signals import mls
from pffdtd.signals import octave
from pffdtd.signals import phase
from pffdtd.signals import pink
from pffdtd.signals import sine


@click.group(help='Signals & DSP.')
def signals():
    pass


signals.add_command(convolution.main)
signals.add_command(fir.main)
signals.add_command(mls.main)
signals.add_command(octave.main)
signals.add_command(phase.main)
signals.add_command(pink.main)
signals.add_command(sine.main)
