# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.filters import dolby
from pffdtd.filters import octave
from pffdtd.filters import phase


@click.group(help='Filters.')
def filters():
    pass


filters.add_command(dolby.main)
filters.add_command(octave.main)
filters.add_command(phase.main)
