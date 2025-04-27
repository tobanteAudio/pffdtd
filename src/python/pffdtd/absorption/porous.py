# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from pffdtd.absorption.admittance import convert_nabs_to_R
from pffdtd.absorption.air import Air, air_density, sound_velocity, wave_number_in_air
from pffdtd.common.plot import plot_styles
from pffdtd.geometry.math import difference_over_sum


def porous_absorber(
    thickness: float,
    flow_resistivity: float,
    frequency: ArrayLike,
    *,
    air_gap: float | None = None,
    angle=0.0,
    temperature=20.0,
    pressure=1.0,
    offset_zeros=False,
) -> np.ndarray:
    density = air_density(pressure, temperature)
    velocity = sound_velocity(temperature)
    air = Air(
        temperature=temperature,
        pressure=pressure,
        density=density,
        velocity=velocity,
        impedance=velocity*density,
        tau_over_c=(2.0*np.pi)/velocity
    )

    minus_i = complex(0.0, -1.0)
    angle_rad = np.deg2rad(angle)
    sin_phi = np.sin(angle_rad)
    cos_phi = np.cos(angle_rad)

    k_air = wave_number_in_air(air, frequency)

    # Characteristic absorber impedance and wave number
    (z_abs, wave_no_abs) = _absorber_props(air, flow_resistivity, frequency)
    wave_no_abs_y = k_air * sin_phi
    wave_no_abs_x = np.sqrt(wave_no_abs**2 - wave_no_abs_y**2)

    # Intermediate term for porous impedance calculation
    porous_wave_no = wave_no_abs * thickness
    cot_porous_wave_no = np.cos(porous_wave_no) / np.sin(porous_wave_no)

    # Impedance at absorber surface
    z_abs_surface = minus_i * z_abs * (wave_no_abs / wave_no_abs_x) * cot_porous_wave_no

    # Calculate absorption coefficient for porous absorber with no air gap
    abs_refl = difference_over_sum((z_abs_surface / air.impedance) * cos_phi, 1.0)
    abs_alpha = _reflectivity_as_alpha(abs_refl)

    if offset_zeros:
        abs_alpha[abs_alpha == 0.0] = np.finfo(abs_alpha.dtype).eps

    if not air_gap:
        return abs_alpha

    # --- AIR GAP ---
    # Angle of propagation within porous layer
    beta_porous = np.rad2deg(np.sin(np.abs(wave_no_abs_y / wave_no_abs)))

    # Impedance values (with air gap)
    # X and Y components of the wave number in the air gap
    wave_no_air_y = wave_no_abs * np.sin(np.deg2rad(beta_porous))
    wave_no_air_x = np.sqrt((k_air * k_air) - (wave_no_air_y * wave_no_air_y))

    # Impedance at top of air gap (after passing through porous absorber)
    temp_imp = k_air * air_gap
    air_gap_z = minus_i * air.impedance * (k_air / wave_no_air_x) * (np.cos(temp_imp) / np.sin(temp_imp))

    # Impedance at top of porous absorber after passing through air gap
    intermediate3 = minus_i * z_abs * cot_porous_wave_no
    abs_air_z = ((air_gap_z * intermediate3) + (z_abs * z_abs)) / (air_gap_z + intermediate3)

    # Absorption coefficient for porous absorber with air gap
    abs_air_refl = difference_over_sum((abs_air_z / air.impedance) * cos_phi, 1.0)
    abs_air_alpha = _reflectivity_as_alpha(abs_air_refl)

    if offset_zeros:
        abs_air_alpha[abs_air_alpha == 0.0] = np.finfo(abs_air_alpha.dtype).eps

    return abs_air_alpha


def _absorber_props(air: Air, flow_resistivity: float, frequency):
    """Characteristic absorber impedance and wave number
    """
    x = (air.density * frequency) / flow_resistivity

    # Complex impedance
    z = air.impedance
    z_abs = z * ((1.0 + 0.0571 * x**-0.754) + 1j * (-0.087 * x**-0.732))

    # Complex wave number
    k = wave_number_in_air(air, frequency)
    k_abs = k * ((1.0 + 0.0978 * x**-0.7) + 1j*(-0.189 * x**-0.595))

    return (z_abs, k_abs)


def _reflectivity_as_alpha(refl):
    """Convert reflectivity to absorption
    """
    alpha = 1.0 - np.abs(refl)**2.0
    alpha[alpha < 0.0] = 0.0
    return alpha


@click.command(name='porous', help='Plot porous absorption properties.')
@click.argument('csv_file', nargs=1, type=click.Path(exists=True))
@click.option('--angle', default=0.0, type=float)
@click.option('--reflection', is_flag=True)
@click.option('--temperature', default=20, type=float)
def main(csv_file, angle, reflection, temperature):
    absorbers = pd.read_csv(csv_file)
    frequency = np.linspace(20, 20_000, 1024*16)

    _, ax = plt.subplots(1, 1)
    ax: Axes = ax

    plt.rcParams.update(plot_styles)
    ax.set_title(csv_file)

    for _, spec in absorbers.iterrows():
        flow_resistivity = spec['flow_resistivity']
        thickness = spec['thickness']
        air_gap = spec['air_gap']
        air_gap = air_gap if not np.isnan(air_gap) else None

        absorber = porous_absorber(
            thickness=thickness,
            flow_resistivity=flow_resistivity,
            frequency=frequency,
            air_gap=air_gap,
            angle=angle,
            temperature=temperature
        )

        label = f"{flow_resistivity:.0f} Pa*s/m² {thickness*100:.0f}cm"
        if air_gap:
            label += f' with {air_gap*100:.0f}cm air gap'

        if reflection:
            ax.semilogx(frequency, convert_nabs_to_R(absorber), label=label)
        else:
            ax.semilogx(frequency, absorber, label=label)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(which='major', linewidth=0.75)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()
    plt.show()
