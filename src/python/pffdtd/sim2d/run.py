# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.sim2d.engine import Engine2D


@click.command(name='run', help='Run simulation.')
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--out', default='out.h5')
@click.option('--video', is_flag=True)
@click.option('--video-color', type=click.Choice(['light-decay', 'black-decay', 'black-norm']))
def main(sim_dir, out, video, video_color):
    engine = Engine2D(sim_dir=sim_dir, out=out, video=video, video_color=video_color)
    engine.run()
    engine.save_output()
