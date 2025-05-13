# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import click
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def max_sound_pressure(SPL_ref, P_max, P_ref=1):
    """Maximum sound pressure level (SPL dB)

    Parameters:
        SPL_ref: Reference sensitivity of the speaker driver
        P_max: Maximum power the speaker can handle
        P_ref: Power used for sensitivity measurement

    Returns:
        SPL_max: Sound pressure level at P_max
    """
    SPL_max = SPL_ref + 10*np.log10(P_max/P_ref)
    return SPL_max


def power_for_target_spl(SPL_target, SPL_ref,  P_ref=1):
    """Required power for given max SPL
    """
    return P_ref * 10**((SPL_target-SPL_ref)/10)


def plot_transducer_power_requirements(df, transducers, SPL_target=108, ax: Axes | None = None):
    if not ax:
        ax = plt.gca()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax.set_title('Power Requirements')
    ax.set_xlabel('SPL [dB]')
    ax.set_ylabel('Power [W]')
    ax.grid(which='minor', color='#222222', linestyle=':', linewidth=0.5)
    ax.vlines(SPL_target, 0, 400, linestyles='--', label=f'Target {SPL_target} dB')

    for name, color in zip(transducers, colors):
        driver = df[df['Name'] == name]
        assert len(driver) == 1

        D_nominal = float(driver['D_nominal'].iloc[0])
        V_ref = float(driver['V_ref'].iloc[0])
        Z_ref = float(driver['Z_ref'].iloc[0])
        P_rms = float(driver['P_rms'].iloc[0])
        P_max = float(driver['P_max'].iloc[0])
        SPL_ref = float(driver['SPL_ref'].iloc[0])

        P_ref = (V_ref**2)/Z_ref
        SPL_rms = max_sound_pressure(SPL_ref, P_rms, P_ref)
        SPL_peak = max_sound_pressure(SPL_ref, P_max, P_ref)
        P_target = power_for_target_spl(SPL_target, SPL_ref, P_ref)

        amp_gain_dB = 25.5
        inV = np.sqrt(P_target*Z_ref)/(10**(amp_gain_dB/20))
        indBu = 20*np.log10(inV/0.7746)

        print(f"- {name}:")
        print(f"    {Z_ref=:.2f} Ohm")
        print(f"    {P_ref=:.2f} W")
        print(f"    {P_rms=:.2f} W")
        print(f"    {P_max=:.2f} W")
        print(f"    {P_target=:.2f} W")
        print(f"    {SPL_ref=:.2f} dB")
        print(f"    {SPL_rms=:.2f} dB")
        print(f"    {SPL_peak=:.2f} dB")
        print('')

        SPL_desired = np.linspace(SPL_ref, SPL_peak, 1024)
        P_required = power_for_target_spl(SPL_desired, SPL_ref, P_ref)

        label = f"{name} {np.nan_to_num(D_nominal):.1f}\" {P_target:.3f} W ({indBu:.2f} dBu)"
        ax.plot(SPL_desired[SPL_desired < SPL_rms], P_required[SPL_desired < SPL_rms], label=label, color=color)
        ax.plot(SPL_desired[SPL_desired >= SPL_rms], P_required[SPL_desired >= SPL_rms], linestyle='--', color=color)
        ax.scatter(SPL_rms, P_rms, color=color)

    ax.legend()


_all_transducers = [
    # 'B&C Speakers 10NW76',
    # 'Ciare 12.00SW',
    # 'Ciare 18.00SW',
    # 'Dayton Audio AMTPRO-4',
    # 'Dayton Audio RSS265HF-8',
    # 'Dayton Audio RSS315HFA-8',
    # 'Dayton Audio RSS315HF-4',
    # 'Dayton Audio RSS315HO-4',
    # 'Dayton Audio RSS390HF-4',
    'Dayton Audio RSS390HO-4',
    'Dayton Audio RSS460HO-4',
    # 'Morel CAT 328-110',
    # 'Morel EM 1308',
    # 'Morel ET 338',
    # 'Morel ET 448',
    # 'Morel ST 1048',
    # 'Morel TSCT 1044',
    # 'Morel TiCW 1058Ft',
    # 'Morel UW 1058',
    # 'Morel UW 1258',
    # 'Mundorf AMT 25CS2.1-R',
    # 'Mundorf AMT 29CM1.1-R',
    # 'Mundorf AMT U160W1.1-R',
    # 'Mundorf AMT U60W1.1-C',
    # 'JBL Selenium D220Ti-8',
    # 'Purifi Audio PTT10.0X04-NAB-01',
    # 'Radian 475PB',
    # 'Radian 745PB',
    # 'Radian 835PB',
    # 'Radian 951PB',
    # 'Radian 950PB',
    # 'SB Acoustics TW29DN-B',
    # 'SB Acoustics TW29BNWG-4',
    # 'SB Acoustics SB12MNRX2-25-4',
    # 'SB Acoustics SB34NRX75-6',
    # 'SB Acoustics SB34NRX75-8',
    # 'Scan-Speak 30W/4558T00',
    # 'Scan-Speak 32W/4878T00',
    # 'Scan-Speak 32W/4878T01',
    # 'Supravox 215 GMF',
    # 'Supravox 285 GMF',
    # 'Supravox 400 GMF',
    # 'Supravox 400-2000 EXC',
    # 'TAD TD-2002',
    # 'TAD TD-4001',
    'TAD TL-1601b',
    'TAD TL-1801',
    # 'Volt Loudspeakers VM527',
    # 'Volt Loudspeakers VM752',
    # 'Volt Loudspeakers RV2501',
    # 'Volt Loudspeakers RV3143',
    # 'Volt Loudspeakers RV3863',
    # 'Volt Loudspeakers RV4564',
    # 'Wavecor TW030WA21',
]


@click.command(name='power', help='Power requirements.')
@click.argument('transducers', nargs=-1, type=str)
@click.option('--database', type=click.Path(exists=True), help='Transducers CSV')
@click.option('--SPL_target', 'SPL_target', default=105, show_default=True)
def main(transducers, database, SPL_target):
    plot_transducer_power_requirements(
        pd.read_csv(database),
        transducers if len(transducers) > 0 else _all_transducers,
        SPL_target=SPL_target,
        ax=plt.gca(),
    )
    plt.show()
