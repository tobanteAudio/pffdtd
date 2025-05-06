# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import matplotlib.pyplot as plt

from pffdtd.common.plot import plot_styles

from pffdtd.absorption.cli import absorption
from pffdtd.analysis.cli import analysis
from pffdtd.diffusor.cli import diffusor
from pffdtd.geometry.cli import geometry
from pffdtd.hpc.cli import hpc
from pffdtd.signals.cli import signals
from pffdtd.sim2d.cli import sim2d
from pffdtd.sim3d.cli import sim3d
from pffdtd.speaker.cli import speaker
from pffdtd.voxelizer.cli import voxelizer


@click.group()
@click.option('--verbose', is_flag=True, help='Print debug output.')
@click.pass_context
def main(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    plt.rcParams.update(plot_styles)


main.add_command(absorption)
main.add_command(analysis)
main.add_command(diffusor)
main.add_command(geometry)
main.add_command(hpc)
main.add_command(signals)
main.add_command(sim2d)
main.add_command(sim3d)
main.add_command(speaker)
main.add_command(voxelizer)
