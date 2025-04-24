# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


import sys

import numpy as np
import pandas as pd


def resonance_frequency(Cms, Mms):
    fs = 1.0/(2*np.pi*np.sqrt(Cms*Mms))
    return fs


def compliance_equivalent_volume(Sd, Cms, rho=1.2, c=343.2):
    Vas = Cms * rho * c**2 * Sd**2
    return Vas


def diaphragm_diameter(Sd):
    return np.sqrt(Sd/np.pi)*2


def electrical_q_factor(Mms, fs, Re, BL):
    return (2*np.pi*fs*Mms*Re)/(BL**2)


def mechanical_q_factor(Mms, Rms, fs):
    return (2*np.pi*fs*Mms)/Rms


def mechanical_resistance(Sd, fs, Qms, Vas, rho=1.2, c=343.2):
    Rms = (rho*c**2*Sd**2)/(2*np.pi*fs*Qms*Vas)
    return Rms


def mechanical_compliance(Sd, Vas, rho=1.2, c=343.2):
    Cms = Vas/(rho * c**2 * Sd**2)
    return Cms


def max_impedance(Qms, Qes, Re):
    Zmax = Re*(1+Qms/Qes)
    return Zmax


def efficiency(fs, Qes, Vas, c=343.2):
    n0 = (4.0 * np.pi**2 * fs**3 * Vas) / (c**3 * Qes)
    return n0


def main():
    # # RSS315HFA-8
    # Cms = 0.00027
    # Mms = 0.194
    # Sd = 0.05067
    # BL = 18
    # Re = 6.5
    # Qms = 2.5

    # # Volt RV3143
    # Cms = 0.00027
    # Mms = 0.076
    # Sd = 0.0473
    # BL = 18
    # Re = 6.1
    # Qms = 5.35

    # # Volt RV3863
    # Cms = 0.000253
    # Mms = 0.119
    # Sd = 0.0760
    # BL = 19.3
    # Re = 5.8
    # Qms = 3.94

    # # Step 1
    # Fs = resonance_frequency(Cms, Mms)

    # # Step 2
    # Vas = compliance_equivalent_volume(Sd, Cms)
    # Dd = diaphragm_diameter(Sd)
    # Qes = electrical_q_factor(Mms, Fs, Re, BL)
    # Rms = mechanical_resistance(Sd, Fs, Qms, Vas)
    # n0 = efficiency(Fs, Qes, Vas)
    # Zmax = max_impedance(Qms, Qes, Re)

    # print(f'{Fs=:.2f} Hz')
    # print(f'{Vas=:.5f} m^3')
    # print(f'{Dd=:.3f} m')
    # print(f'{Qes=:.3f}')
    # print(f'{Rms=:.3f} kg/s')
    # print(f'{Zmax=:.3f} Ohm')
    # print(f'n0={n0*100:.4f} %')

    drivers = pd.read_csv(sys.argv[1], encoding='utf-8')
    drivers = drivers[drivers['Type'] == 'Midrange']
    drivers = drivers.drop(columns=['fmin', 'fmax', 'f1', 'f2', 'F3_sealed', 'F3_ported', 'Volume_sealed', 'Volume_ported'])

    # drivers['n0'] = efficiency(drivers['fs'], drivers['Qes'], drivers['Vas']/1000)*100
    # drivers['Cms_'] = mechanical_compliance(drivers['Sd'], drivers['Vas']/1000)
    # drivers['Zmax'] = max_impedance(drivers['Qms'], drivers['Qes'], drivers['Re'])
    # drivers['Error'] = (drivers['Cms_']-drivers['Cms'])/drivers['Cms']*100
    # drivers['Cms_'] = drivers['Cms_'].round(6)
    # drivers['error'] = drivers['Zmax_prime']-drivers['Zmax']
    # print(drivers[['Name', 'Zmax', 'Zmax_prime', 'error']].to_string(na_rep='', index=False))

    # print(drivers[['Name', 'n0_prime']].to_string(na_rep='', index=False))
    drivers.sort_values('Price', inplace=True, ascending=False)
    print(drivers.to_string(na_rep='', index=False))


if __name__ == '__main__':
    main()
