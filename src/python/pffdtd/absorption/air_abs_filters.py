# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""This is a testbench for different air absorption filters
"""
import matplotlib.pyplot as plt
import numpy as np

from pffdtd.geometry.math import iround
from pffdtd.absorption.air import (
    apply_modal_filter,
    apply_ola_filter,
    apply_visco_filter,
)


def main():
    Tc = 20
    rh = 60
    fs = 48e3

    # single channel test
    t_end = 1
    Nt0 = iround(t_end*fs)
    tx = np.arange(Nt0)/fs
    td = 0.02
    tau = t_end/(6*np.log(10))
    x = np.exp(-(tx-td)/tau)*(2*np.random.random_sample(Nt0)-1)
    x[:int(td*fs)] = 0

    y1 = apply_visco_filter(x, fs, Tc, rh)
    y2 = apply_modal_filter(x, fs, Tc, rh)
    y3 = apply_ola_filter(x, fs, Tc, rh)

    ty1 = np.arange(0, y1.shape[-1])/fs
    ty2 = np.arange(0, y2.shape[-1])/fs
    ty3 = np.arange(0, y3.shape[-1])/fs

    plt.plot(tx, x.T, linestyle='-', color='b', label='orig')
    plt.plot(ty1, y1.T, linestyle='-', color='g', label='stokes')
    plt.plot(ty2, y2.T, linestyle='-', color='r', label='modal')
    plt.plot(ty3, y3.T, linestyle='-', color='y', label='OLA')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
