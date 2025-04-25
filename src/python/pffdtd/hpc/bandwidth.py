# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import pandas as pd


@click.command(name='bandwidth', help='RAM Bandwidth')
def main():
    speed = 5600
    pumps = 2
    width = 64
    channels = 12

    width /= 8
    mhz_speed = speed / pumps
    transfers = speed * 1_000_000
    channel_bw = transfers * width / 1e9
    cpu_bw = channel_bw * channels

    print(f'{mhz_speed=:.1f} Mhz')
    print(f'transfers={transfers/1e6:.1f} MT/s')
    print(f'{channel_bw=:.1f} GB/s')
    print('')

    cpu_ram = 64*12
    cpu_it = cpu_bw/cpu_ram
    print(f'{cpu_bw=:.1f} GB/s')
    print(f'{cpu_ram=:.0f} GB')
    print(f'{cpu_it=:.2f} it/s')
    print('')

    gpu_ram = 256
    gpu_bw = 6000
    gpu_it = gpu_bw/gpu_ram
    print(f'{gpu_bw=:.2f} GB/s')
    print(f'{gpu_ram=:.0f} GB')
    print(f'{gpu_it=:.2f} it/s')
    print('')

    print(f'gpu_it/cpu_it={gpu_it/cpu_it:.2f}')

    mi325x_vram = 256
    mi325x_bandwidth = 6000
    mi325x_iter = mi325x_bandwidth/mi325x_vram

    h200_vram = 141
    h200_bandwidth = 4800
    h200_iter = h200_bandwidth/h200_vram

    epyc_ram = 32*24
    epyc_bandwidth = cpu_bw*2
    epyc_iter = epyc_bandwidth/epyc_ram

    devices = [
        {
            'Name': 'NVIDIA H200',
            'Type': 'GPU',
            'Device GB': h200_vram,
            'Device GB/s': h200_bandwidth,
            'Devices': 8,
            'System GB': h200_vram*8,
            'System GB/s': h200_bandwidth*8,
            'System Cost': 300e3,
            'Iterations': h200_iter,
        },
        {
            'Name': 'AMD MI325x',
            'Type': 'GPU',
            'Device GB': mi325x_vram,
            'Device GB/s': mi325x_bandwidth,
            'Devices': 8,
            'System GB': mi325x_vram*8,
            'System GB/s': mi325x_bandwidth*8,
            'System Cost': 250e3,
            'Iterations': mi325x_iter,
        },
        {
            'Name': 'AMD EPYC',
            'Type': 'CPU',
            'Device GB': epyc_ram/2,
            'Device GB/s': epyc_bandwidth/2,
            'Devices': 2,
            'System GB': epyc_ram,
            'System GB/s': epyc_bandwidth,
            'System Cost': 27e3,
            'Iterations': epyc_iter,
        },
        {
            'Name': 'Mac Studio M3 Ultra',
            'Type': 'CPU',
            'Device GB': 512,
            'Device GB/s': 800,
            'Devices': 1,
            'System GB': 512,
            'System GB/s': 800,
            'System Cost': 11e3,
            'Iterations': 800/512,
        },
    ]

    df = pd.DataFrame.from_records(devices)
    df['Memory/Cost'] = df['System GB']/(df['System Cost']/1e3)
    print(df.to_markdown(index=False))
