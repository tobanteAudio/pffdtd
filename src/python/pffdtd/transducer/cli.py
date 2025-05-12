# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click

from pffdtd.transducer import distortion
from pffdtd.transducer import diy
from pffdtd.transducer import horn
from pffdtd.transducer import ts


@click.group(help='Transducers.')
def transducer():
    pass


transducer.add_command(distortion.main)
transducer.add_command(diy.main)
transducer.add_command(horn.main)
transducer.add_command(ts.main)
