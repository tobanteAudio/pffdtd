# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click

from pffdtd.speaker import distortion
from pffdtd.speaker import diy
from pffdtd.speaker import horn
from pffdtd.speaker import ts


@click.group(help='Speaker.')
def speaker():
    pass


speaker.add_command(distortion.main)
speaker.add_command(diy.main)
speaker.add_command(horn.main)
speaker.add_command(ts.main)
