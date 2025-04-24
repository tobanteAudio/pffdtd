# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click

from pffdtd.hpc import bandwidth
from pffdtd.hpc import cluster
from pffdtd.hpc import precision


@click.group(help='HPC.')
def hpc():
    pass


hpc.add_command(bandwidth.main)
hpc.add_command(cluster.main)
hpc.add_command(precision.main)
