# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.analysis import calibration
from pffdtd.analysis import localization
from pffdtd.analysis import response
from pffdtd.analysis import rew
from pffdtd.analysis import room_modes
from pffdtd.analysis import rt60
from pffdtd.analysis import spectrogram
from pffdtd.analysis import summary


@click.group(help='Analysis.')
def analysis():
    pass


analysis.add_command(calibration.main)
analysis.add_command(localization.main)
analysis.add_command(response.main)
analysis.add_command(rew.main)
analysis.add_command(room_modes.main)
analysis.add_command(rt60.main)
analysis.add_command(spectrogram.main)
analysis.add_command(summary.main)
