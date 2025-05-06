# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import numpy as np


def volts_to_dB(V, Vref=1.0):
    return 20*np.log10(V/Vref)


def dBV_to_volts(dBV):
    return 10**(dBV/20)


def dBV_to_dBu(dBV, Vref=0.7746):
    return dBV + 20*np.log10(1/Vref)


def dBu_to_dBV(dBu, Vref=0.7746):
    return dBu - 20*np.log10(1/Vref)
