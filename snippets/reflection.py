# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
"""
Two-path comb filtering (single reflection) calculator.

Given direct path d1 and reflected path d2, the response magnitude is:
|H(f)| = sqrt(1 + r^2 + 2 r cos(2π f Δt + φ))
where:
  r   = A_reflected / A_direct (pressure amplitude ratio)
  Δt  = (d2 - d1) / c
  φ   = extra phase term; use φ=π for polarity inversion (180°)

Notch/peak frequencies:
- If φ = 0 (no inversion):
    notches: f = (2n+1)/(2Δt)
    peaks:   f = n/Δt
- If φ = π (inversion):
    notches: f = n/Δt
    peaks:   f = (2n+1)/(2Δt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class CombResult:
    d1_m: float
    d2_m: float
    c_m_s: float
    delta_d_m: float
    delta_t_s: float
    spacing_hz: float
    r_linear: float
    reflection_db: float
    inversion: bool
    notches_hz: List[float]
    peaks_hz: List[float]
    h_min_lin: float
    h_max_lin: float
    h_min_db: float
    h_max_db: float
    peak_to_notch_db: float


def db_to_linear_amp(db: float) -> float:
    """Convert dB (amplitude/pressure) to linear amplitude ratio."""
    return 10.0 ** (db / 20.0)


def linear_amp_to_db(x: float, floor: float = 1e-12) -> float:
    """Convert linear amplitude to dB, with a floor to avoid log(0)."""
    x = max(abs(x), floor)
    return 20.0 * math.log10(x)


def generate_series(delta_t: float, f_max: float, inversion: bool) -> Tuple[List[float], List[float]]:
    """
    Generate notch and peak frequency lists up to f_max.
    """
    notches: List[float] = []
    peaks: List[float] = []

    if delta_t <= 0:
        return notches, peaks

    # Helper lambdas for the two cases
    if not inversion:
        def notch_fn(n): return (2 * n + 1) / (2.0 * delta_t)
        def peak_fn(n): return n / delta_t
    else:
        def notch_fn(n): return n / delta_t
        def peak_fn(n): return (2 * n + 1) / (2.0 * delta_t)

    # Notches: start at n=0 typically (except inversion case includes 0 Hz)
    n = 0
    while True:
        f = notch_fn(n)
        if f <= 0:
            n += 1
            continue
        if f > f_max:
            break
        notches.append(f)
        n += 1

    # Peaks: include n=1.. for printing (n=0 is DC)
    n = 1
    while True:
        f = peak_fn(n)
        if f <= 0:
            n += 1
            continue
        if f > f_max:
            break
        peaks.append(f)
        n += 1

    return notches, peaks


def magnitude_response(f_hz: float, r: float, delta_t: float, inversion: bool) -> float:
    """
    |H(f)| = sqrt(1 + r^2 + 2 r cos(2π f Δt + φ))
    φ = 0   no inversion
    φ = π   inversion
    """
    phi = math.pi if inversion else 0.0
    return math.sqrt(1.0 + r * r + 2.0 * r * math.cos(2.0 * math.pi * f_hz * delta_t + phi))


def compute_comb(
    d1_m: float = 3.0,
    d2_m: float = 7.0,
    c_m_s: float = 343.0,
    reflection_db: float = -6.0,
    inversion: bool = False,
    f_max: float = 1000.0,
) -> CombResult:
    """
    Compute key comb filtering quantities.
    reflection_db is amplitude/pressure dB relative to direct (e.g., -6 dB => r≈0.5).
    """
    delta_d = d2_m - d1_m
    if delta_d <= 0:
        raise ValueError('Reflected path must be longer than direct path (d2_m > d1_m).')

    delta_t = delta_d / c_m_s
    spacing = 1.0 / delta_t

    r = db_to_linear_amp(reflection_db)

    notches, peaks = generate_series(delta_t=delta_t, f_max=f_max, inversion=inversion)

    # Global min/max for simple two-path model:
    # |H|min = |1 - r| (no inversion) or |1 - r| still holds at the destructive condition,
    # |H|max = 1 + r
    # (These remain true even with inversion; inversion just shifts where they occur.)
    h_min = abs(1.0 - r)
    h_max = 1.0 + r

    h_min_db = linear_amp_to_db(h_min)
    h_max_db = linear_amp_to_db(h_max)
    peak_to_notch_db = h_max_db - h_min_db

    return CombResult(
        d1_m=d1_m,
        d2_m=d2_m,
        c_m_s=c_m_s,
        delta_d_m=delta_d,
        delta_t_s=delta_t,
        spacing_hz=spacing,
        r_linear=r,
        reflection_db=reflection_db,
        inversion=inversion,
        notches_hz=notches,
        peaks_hz=peaks,
        h_min_lin=h_min,
        h_max_lin=h_max,
        h_min_db=h_min_db,
        h_max_db=h_max_db,
        peak_to_notch_db=peak_to_notch_db,
    )


def main() -> None:
    # --- User-configurable inputs ---
    d1_m = 3.000
    d2_m = 3.803
    c_m_s = 343.0          # speed of sound (m/s); adjust for temperature if desired
    absorption_db = 0.0   # amplitude/pressure dB of reflection relative to direct
    inversion = False      # True if reflection is polarity-inverted (adds 180°)
    f_max = 1000.0         # list notches/peaks up to this frequency
    do_plot = True         # set False to skip plotting
    distance_attenuation = 20*math.log10(d1_m/d2_m)
    reflection_db = distance_attenuation+absorption_db
    # --------------------------------

    res = compute_comb(
        d1_m=d1_m,
        d2_m=d2_m,
        c_m_s=c_m_s,
        reflection_db=reflection_db,
        inversion=inversion,
        f_max=f_max,
    )

    print('=== Two-path comb filter ===')
    print(f"Direct path:          {res.d1_m:.3f} m")
    print(f"Reflected path:       {res.d2_m:.3f} m")
    print(f"Path diff Δd:         {res.delta_d_m:.3f} m")
    print(f"Delay Δt:             {res.delta_t_s*1000:.3f} ms")
    print(f"Comb spacing:         {res.spacing_hz:.3f} Hz (1/Δt)")
    print(f"Distance attenuation: {distance_attenuation} dB")
    print(f"Absorption:           {absorption_db} dB")
    print(f"Reflection level:     {res.reflection_db:.2f} dB  (r = {res.r_linear:.4f} linear amp)")
    print(f"Inversion:            {res.inversion}")
    print()
    print('Expected extrema (ideal two-path model):')
    print(f"  Min |H|: {res.h_min_lin:.6f}  ({res.h_min_db:.2f} dB)")
    print(f"  Max |H|: {res.h_max_lin:.6f}  ({res.h_max_db:.2f} dB)")
    print(f"  Peak-to-notch range: {res.peak_to_notch_db:.2f} dB")
    print()

    def fmt_list(name: str, freqs: List[float], limit: int = 20) -> None:
        print(f"{name} (Hz) up to {f_max:.0f} Hz:")
        shown = freqs[:limit]
        print('  ' + ', '.join(f"{f:.1f}" for f in shown) + (' ...' if len(freqs) > limit else ''))
        print()

    fmt_list('Notches', res.notches_hz)
    fmt_list('Peaks', res.peaks_hz)

    # Optional plot
    if do_plot:
        if plt is None:
            print('matplotlib not available; skipping plot.')
            return

        n_points = 4000
        f = [i * (f_max / n_points) for i in range(n_points + 1)]
        mag = [magnitude_response(fi, res.r_linear, res.delta_t_s, res.inversion) for fi in f]
        mag_db = [linear_amp_to_db(m) for m in mag]

        plt.figure()
        plt.semilogx(f, mag_db)
        plt.xlim(10, f_max)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dBr)')
        plt.title('Two-path comb filtering')
        plt.grid(which='both')
        plt.show()


if __name__ == '__main__':
    main()
