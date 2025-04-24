# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.analysis import response
from pffdtd.analysis import room_modes
from pffdtd.analysis import signals
from pffdtd.analysis import t60
from pffdtd.analysis import waterfall


@click.group(help='Analysis.')
def analysis():
    pass


analysis.add_command(response.main)
analysis.add_command(room_modes.main)
analysis.add_command(signals.main)
analysis.add_command(t60.main)
analysis.add_command(waterfall.main)
