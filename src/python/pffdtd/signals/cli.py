# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.signals import dolby
from pffdtd.signals import mls
from pffdtd.signals import octave
from pffdtd.signals import phase
from pffdtd.signals import pink
from pffdtd.signals import sine


@click.group(help='signals.')
def signals():
    pass


signals.add_command(dolby.main)
signals.add_command(mls.main)
signals.add_command(octave.main)
signals.add_command(phase.main)
signals.add_command(pink.main)
signals.add_command(sine.main)
