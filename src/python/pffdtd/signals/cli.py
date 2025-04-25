# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.signals import mls
from pffdtd.signals import pink


@click.group(help='signals.')
def signals():
    pass


signals.add_command(mls.main)
signals.add_command(pink.main)
