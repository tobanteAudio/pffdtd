# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import numpy as np
import pandas as pd


def resonance_frequency(Cms, Mms):
    fs = 1.0/(2*np.pi*np.sqrt(Cms*Mms))
    return fs


def compliance_equivalent_volume(Sd, Cms, rho=1.2, c=343.2):
    Vas = Cms * rho * c**2 * Sd**2
    return Vas


def diaphragm_diameter(Sd):
    Dd = np.sqrt(Sd/np.pi)*2
    return Dd


def electrical_q_factor(Mms, fs, Re, BL):
    Qes = (2*np.pi*fs*Mms*Re)/(BL**2)
    return Qes


def mechanical_q_factor(Mms, Rms, fs):
    Qms = (2*np.pi*fs*Mms)/Rms
    return Qms


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


@click.command(name='ts', help='Thiele/Small parameters')
@click.argument('drivers_csv', nargs=1, type=click.Path(exists=True))
@click.argument('driver_type', type=str, default='')
def main(drivers_csv, driver_type):
    drivers = pd.read_csv(drivers_csv, encoding='utf-8')
    drivers = drivers.drop(columns=['fmin', 'fmax', 'f1', 'f2', 'F3_sealed', 'F3_ported', 'Volume_sealed', 'Volume_ported'])
    if driver_type:
        drivers = drivers[drivers['Type'] == driver_type]

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
