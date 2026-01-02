# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import math


def required_digital_level(
    target_spl_db,
    speaker_sensitivity_db,
    amp_gain_db,
    interface_max_dbu,
    distance_m=1.0,
    speaker_impedance_ohm=8.0
):
    """
    Returns the required digital signal level in dBFS (RMS) to achieve
    the desired SPL at 1 meter.

    Parameters:
        target_spl_db (float): desired SPL at 1 m
        speaker_sensitivity_db (float): dB SPL @ 2.83V (≈1W) @ 1m
        amp_gain_db (float): amplifier voltage gain in dB
        interface_max_dbu (float): interface maximum output level in dBu
        distance_m (float): listening distance in meters (default 1 m)
        speaker_impedance_ohm (float): speaker nominal impedance
    """

    # === Step 1: Required SPL relative to speaker sensitivity ===
    spl_needed_increase = target_spl_db - speaker_sensitivity_db

    # Required power factor relative to 1 W
    power_ratio = 10 ** (spl_needed_increase / 10)

    # Required output power
    p_ref = (2.83**2) / speaker_impedance_ohm
    required_power_w = p_ref * power_ratio

    # Required speaker voltage
    required_spk_v = math.sqrt(required_power_w * speaker_impedance_ohm)

    # === Step 2: Work backwards through amplifier ===
    amp_voltage_gain = 10 ** (amp_gain_db / 20)
    required_amp_input_v = required_spk_v / amp_voltage_gain

    # === Step 3: Compare with interface maximum ===
    interface_max_v = 0.775 * (10 ** (interface_max_dbu / 20))

    # Digital level ratio
    ratio = required_amp_input_v / interface_max_v

    # Convert to dBFS level (negative number)
    required_dbfs = 20 * math.log10(ratio)

    return {
        'required_power_w': required_power_w,
        'required_speaker_voltage_v': required_spk_v,
        'required_amp_input_voltage_v': required_amp_input_v,
        'interface_max_voltage_v': interface_max_v,
        'digital_level_dbfs': required_dbfs,
    }


# === Example using your numbers ===
result = required_digital_level(
    target_spl_db=105,
    speaker_sensitivity_db=98,
    amp_gain_db=15,
    interface_max_dbu=16,
    speaker_impedance_ohm=4
)

for k, v in result.items():
    print(f"{k}: {v:.3f}")
