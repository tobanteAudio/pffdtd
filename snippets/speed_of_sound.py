# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import math
from typing import Optional

R_UNIVERSAL = 8.31446261815324  # J/(mol·K)

# Molar masses (kg/mol)
M_DRY_AIR = 28.96546e-3
M_WATER_VAPOR = 18.01528e-3

# Common acoustics approximation for gamma (ratio of specific heats)
GAMMA_AIR = 1.4


def pressure_from_elevation_standard_atmosphere(elevation_m: float) -> float:
    """
    Estimate ambient pressure (Pa) from elevation (m) using the ISA troposphere
    model (valid up to ~11 km). Adequate for most near-surface acoustics.

    Assumptions:
      - Sea-level pressure p0 = 101325 Pa
      - Sea-level temperature T0 = 288.15 K
      - Lapse rate L = 0.0065 K/m
    """
    p0 = 101325.0  # Pa
    T0 = 288.15    # K
    L = 0.0065     # K/m
    g0 = 9.80665   # m/s^2
    R_spec_dry = 287.05  # J/(kg·K), for dry air in ISA

    h = float(elevation_m)
    if h < 0:
        # Below sea level: use same formula (still reasonable for modest depths)
        h = h

    if h <= 11000.0:
        return p0 * (1.0 - (L * h) / T0) ** (g0 / (R_spec_dry * L))
    else:
        # Above 11 km: keep it simple; extend with isothermal layer.
        # If you need stratosphere accuracy, implement full ISA layers.
        T11 = T0 - L * 11000.0
        p11 = p0 * (1.0 - (L * 11000.0) / T0) ** (g0 / (R_spec_dry * L))
        g0 = 9.80665
        h_diff = h - 11000.0
        return p11 * math.exp(-g0 * h_diff / (R_spec_dry * T11))


def saturation_vapor_pressure_buck_pa(T_C: float) -> float:
    """
    Buck (1981) saturation vapor pressure over water.
    Returns e_s in Pa for temperature in °C.

    Valid for typical atmospheric temperatures; excellent for acoustics use.
    """
    # Buck formula gives kPa; convert to Pa
    es_kpa = 0.61121 * math.exp((18.678 - (T_C / 234.5)) * (T_C / (257.14 + T_C)))
    return es_kpa * 1000.0


def speed_of_sound_moist_air(
    T_C: float,
    relative_humidity: float,
    elevation_m: float = 0.0,
    pressure_pa: Optional[float] = None,
    gamma: float = GAMMA_AIR,
) -> float:
    """
    Version 2: Ideal-gas mixture model using T + RH (+ elevation for pressure).
    Returns speed of sound c in m/s.

    Inputs:
      - T_C: temperature in °C
      - relative_humidity: RH in [0..1] (e.g., 0.55 for 55%)
      - elevation_m: meters above sea level (used only if pressure_pa is None)
      - pressure_pa: optionally provide measured ambient pressure in Pa
      - gamma: ratio of specific heats (default 1.4). Keeping it constant is a
               common engineering approximation.

    Model:
      - Compute vapor partial pressure e = RH * e_s(T) (Buck)
      - Mole fraction of water vapor x_w ≈ e / p
      - Mixture molar mass M = (1-x_w)*M_dry + x_w*M_water
      - R_mix = R_universal / M
      - c = sqrt(gamma * R_mix * T_K)
    """
    # Basic input hygiene
    rh = float(relative_humidity)
    rh = max(0.0, min(1.0, rh))

    T_K = float(T_C) + 273.15

    p = float(pressure_pa) if pressure_pa is not None else pressure_from_elevation_standard_atmosphere(elevation_m)

    # Vapor pressure
    e_s = saturation_vapor_pressure_buck_pa(T_C)
    e = rh * e_s

    # Prevent non-physical cases (e.g., supersaturation relative to total pressure)
    # In typical conditions e << p. Clamp for numerical safety.
    e = min(e, 0.99 * p)

    # Water vapor mole fraction (ideal gas)
    x_w = e / p

    # Mixture molar mass and gas constant
    M_mix = (1.0 - x_w) * M_DRY_AIR + x_w * M_WATER_VAPOR
    R_mix = R_UNIVERSAL / M_mix  # J/(kg·K)

    # Speed of sound
    c = math.sqrt(float(gamma) * R_mix * T_K)
    return c


if __name__ == '__main__':
    # Example usage:
    T_C = 15.0
    RH = 0.5
    elev = 34
    pressure = 1022*100

    print(pressure_from_elevation_standard_atmosphere(elev))

    for Tc in range(0, 31, 2):
        print('-----')
        # print(f'c ≈ {343.2*math.sqrt(Tc/20):.3f}m/s sim')
        c = 331.3+0.606*Tc
        print(f'c ≈ {c:.3f}m/s estimate 20Hz={c/20:.3f}m')

        c = speed_of_sound_moist_air(Tc, RH, elevation_m=elev)
        print(f"c ≈ {c:.3f}m/s at T={Tc}°C, RH={RH*100:.0f}%, elevation={elev}m 20Hz={c/20:.3f}m")

        c = speed_of_sound_moist_air(Tc, RH, pressure_pa=pressure)
        print(f"c ≈ {c:.3f}m/s at T={Tc}°C, RH={RH*100:.0f}%, pressure={pressure/100:.0f} hPa 20Hz={c/20:.3f}m")
