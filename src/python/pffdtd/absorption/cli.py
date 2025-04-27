# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click

from pffdtd.absorption import admittance
from pffdtd.absorption import air
from pffdtd.absorption import porous


@click.group(help='Absorption.')
def absorption():
    pass


absorption.add_command(admittance.main)
absorption.add_command(air.main)
absorption.add_command(porous.main)
