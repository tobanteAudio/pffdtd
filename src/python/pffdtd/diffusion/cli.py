# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.diffusion import diffusor
from pffdtd.diffusion import measurement


@click.group(help='Diffusor.')
def diffusion():
    pass


diffusion.add_command(diffusor.main)
diffusion.add_command(measurement.main)
