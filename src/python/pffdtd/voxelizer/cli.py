# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.voxelizer import vox_grid
from pffdtd.voxelizer import vox_scene


@click.group(help='Voxelizer.')
def voxelizer():
    pass


voxelizer.add_command(vox_grid.main)
voxelizer.add_command(vox_scene.main)
