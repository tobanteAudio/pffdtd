# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.dsp import dolby
from pffdtd.dsp import mls
from pffdtd.dsp import octave
from pffdtd.dsp import phase
from pffdtd.dsp import pink
from pffdtd.dsp import sine


@click.group(help='DSP.')
def dsp():
    pass


dsp.add_command(dolby.main)
dsp.add_command(mls.main)
dsp.add_command(octave.main)
dsp.add_command(phase.main)
dsp.add_command(pink.main)
dsp.add_command(sine.main)
