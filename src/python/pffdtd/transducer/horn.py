# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import matplotlib.pyplot as plt
import numpy as np


def exponential_horn(s0, sL, L, npts=100):
    """https://www.quarter-wave.com/Horns/Horn_Physics.pdf
    """
    m = np.log(sL/s0)/L
    x = np.linspace(0.0, L, num=npts, endpoint=True)
    s = s0 * np.exp(m*x)
    return s, x


@click.command(name='horn', help='Acoustic horns.')
def main():
    s, x = exponential_horn(2.54, 10.0, 30.0)
    print(s)
    print(x)

    plt.plot(x, +s)
    plt.plot(x, -s)
    plt.xlabel('L [cm]')
    plt.ylabel('S [cm]')
    plt.grid(which='both')
    plt.show()
