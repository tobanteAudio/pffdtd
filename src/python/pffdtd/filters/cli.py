# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.filters import dolby


@click.group(help='Filters.')
def filters():
    pass


filters.add_command(dolby.main)
