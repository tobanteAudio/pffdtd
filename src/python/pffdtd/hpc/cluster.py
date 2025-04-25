# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import numpy as np

from pffdtd.sim3d.constants import SimConstants
from pffdtd.voxelizer.cart_grid import CartGrid


def print_cluster_stats(
    required_ram,
    title,
    device_name,
    device_memory,
    devices_per_node,
    node_tdp,
    node_price,
    node_rental_price
):
    device_count = np.ceil(required_ram/device_memory)
    node_count = np.ceil(device_count/devices_per_node)

    cluster_power = node_count*node_tdp/1e3
    cluster_price = node_count*node_price

    energy_cost = 0.15
    data_center_pue = 1.3
    daily_energy_cost = cluster_power*energy_cost*24*data_center_pue

    cluster_rental_revenue = (node_count*24*node_rental_price)-daily_energy_cost
    cluster_rental_roi = np.ceil(cluster_price/(cluster_rental_revenue))

    print(f'--- {title} ---')
    print(f'ram/dev  = {device_memory:.0f} GB')
    print(f'ram/node = {device_memory*devices_per_node:.0f} GB')
    print(f'device   = {device_count:.0f} x {device_name}')
    print(f'server   = {node_count:.0f} with {devices_per_node}x{device_name}')
    print(f'price    = ${cluster_price/1e6:.3f} Million')
    print(f'energy   = {cluster_power:.3f} kW - ${daily_energy_cost*365/1e3:.2f} Thousand/year')
    print(f'rental   = ${cluster_rental_revenue*365/1e6:.3f} Million/year')
    print(f'roi      = {cluster_rental_roi} days')
    print('')


def missing_notes(fmax):
    note = np.arange(200)
    freqs = 440.0 * 2**((note-69)/12)
    audible_range = freqs[(freqs > 20.0) & (freqs < 20_000.0)]
    sim_range = freqs[(freqs > 20.0) & (freqs < fmax)]

    print('--- AUDIBLE ---')
    print(len(audible_range))
    print(audible_range[0])
    print(audible_range[-1])
    print('')

    print('--- SIM ---')
    print(len(sim_range))
    print(sim_range[0])
    print(sim_range[-1])
    print('')

    print('--- MISSING ---')
    print(f'{len(audible_range)-len(sim_range)}/{len(audible_range)}')
    print(f'{(len(audible_range)-len(sim_range))/len(audible_range)*100:.2f}%')
    print('')


@click.command(name='cluster', help='Multi-node clusters')
def main():
    bmin = [0, 0, 0]

    bmax = [11.2, 7.3, 3.2]
    bmax = [12, 8, 6]
    bmax = [9, 7, 5]
    bmax = [6, 3.65, 3.12]  # Tobi Office
    bmax = [17, 15, 9]
    bmax = [7, 8, 3.2]
    fmax = 20_000.0
    ppw = 10.5
    fcc = False

    # bmax = [4000, 2500, 750]  # Airport
    # bmax = [300, 250, 75]  # Generic Stadium
    # bmax = [100, 100, 60]  # Generic Arena
    # bmax = [315, 280, 133]  # Wembley
    # fmax = 20_000.0
    # ppw = 3.21
    # fcc = False

    constants = SimConstants(20, 50, fmax=fmax, PPW=ppw, fcc=fcc)
    grid = CartGrid(constants.h, 3.0, bmin, bmax, fcc)
    ram_f32, ram_f64 = grid.memory_requirements()

    missing_notes(fmax)

    print('--- GENERAL ---')
    print(f'float32 = {ram_f32:.3f} GB / {ram_f32/1e3:.3f} TB')
    print(f'float64 = {ram_f64:.3f} GB / {ram_f64/1e3:.3f} TB')
    print('')

    print_cluster_stats(ram_f64, 'NVIDIA', 'H200', 141, 8, 700*8+2500, 300000, 30)
    print_cluster_stats(ram_f64, 'AMD', 'MI325X', 256, 8, 1000*8+2500, 20000*8+10000, 20)
    print_cluster_stats(ram_f64, 'CPU', 'EPYC 9005F', 64*12, 2, 2600, 28000, 6.0)
