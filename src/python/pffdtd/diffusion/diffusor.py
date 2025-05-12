# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import numpy as np


def diffusor_bandwidth(max_depth, well_width, c=343.0):
    fmin = c/(max_depth*2)
    fmax = c/(well_width*2)
    return fmin, fmax


def diffusor_dimensions(fmin, fmax, c=343.0):
    assert fmax > fmin
    max_depth = c/(fmin*2)
    well_width = c/(fmax*2)
    return max_depth, well_width


def quadratic_residue_diffusor(prime, depth=None):
    n = np.mod(np.arange(0, prime, 1)**2, prime)
    if depth:
        n = n / np.max(n)
        return n*depth
    return n


def primitive_root_diffusor(prime, g=None, depth=None):
    if g:
        assert _is_primitive_root(prime, g)
    else:
        g = _find_primitive_root(prime)
    n = np.mod(g**np.arange(0, prime-1), prime)
    if depth:
        n = n / np.max(n)
        n *= depth
    return n, g


def _is_primitive_root(prime, g):
    if g in (0, 1):
        return False
    powers = set()
    for i in range(1, prime):
        powers.add(pow(g, i, prime))
    return len(powers) == prime - 1


def _find_primitive_root(prime):
    for g in range(2, prime):
        if _is_primitive_root(prime, g):
            return g
    return None


@click.command(name='diffusor', help='Design diffusors QRD & PRD.')
def main():
    n = 7
    c = 343
    well_width = 0.05
    design_frequency = 500

    design_wavelength = c/design_frequency
    design_depth = design_wavelength/2
    plate_frequency = design_frequency*n
    fmin = design_frequency/2
    fmax = c/(well_width*2)
    seat_distance = design_wavelength*3

    qrd = quadratic_residue_diffusor(n, depth=design_depth)
    prd = primitive_root_diffusor(n, depth=design_depth)

    print(f"prime     = {n}")
    print(f"width     = {well_width*100:.2f} cm")
    print(f"depth     = {design_depth*100:.2f} cm")
    print(f"seat      = {seat_distance*100:.2f} cm")
    print('')

    print(f"scatter   = {fmin:.2f} Hz")
    print(f"diffuse   = {design_frequency:.2f} Hz")
    print(f"HF cutoff = {fmax:.2f} Hz")
    print(f"plate     = {plate_frequency:.2f} Hz")
    print('')

    print(f"QRD     = {np.round(qrd*100, 2)} cm")
    print(f"PRD     = {np.round(prd[0]*100, 2)} cm")
    print(f"max depth = {np.max(qrd)*100:.2f} cm")
