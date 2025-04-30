# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click

from pffdtd.geometry import box
from pffdtd.geometry import tri_box
from pffdtd.geometry import tri_ray


@click.group(help='Geometry.')
def geometry():
    pass


geometry.add_command(box.main)
geometry.add_command(tri_box.main)
geometry.add_command(tri_ray.main)
