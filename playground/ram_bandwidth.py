# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

def main():
    speed = 6400
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

    cpu_ram = 32*12
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

    mi300x_vram = 192
    mi300x_bandwidth = 5300
    mi300x_iter = mi300x_bandwidth/mi300x_vram

    h200_vram = 141
    h200_bandwidth = 4800
    h200_iter = h200_bandwidth/h200_vram

    epyc_ram = 64*24
    epyc_bandwidth = 576*2
    epyc_iter = epyc_bandwidth/epyc_ram

    print(f'AMD-MI300x-VRAM:    {mi300x_vram:.3f} GB')
    print(f'AMD-MI300x-SYS-RAM: {mi300x_vram*8:.3f} GB')
    print(f'AMD-MI300x-SYS-BW:  {mi300x_bandwidth*8:.3f} GB/s')
    print(f'AMD-MI300x-IT:      {mi300x_iter:.3f} / s')
    print('')

    print(f'NVIDIA-H200-VRAM:    {h200_vram:.3f} GB')
    print(f'NVIDIA-H200-SYS-RAM: {h200_vram*8:.3f} GB')
    print(f'NVIDIA-H200-SYS-BW:  {h200_bandwidth*8:.3f} GB/s')
    print(f'NVIDIA-H200-IT:      {h200_iter:.3f} / s')
    print('')

    print(f'EPYC-RAM:    {epyc_ram:.3f} GB')
    print(f'EPYC-SYS-BW: {epyc_bandwidth:.3f} GB/s')
    print(f'EPYC-IT:     {epyc_iter:.3f} / s')
    print('')

    print(f'DIFF-H200-EPYC-SYS-BW: {(h200_bandwidth*8)/epyc_bandwidth:.2f}x')
    print(f'DIFF-MI300x-EPYC-SYS-BW: {(mi300x_bandwidth*8)/epyc_bandwidth:.2f}x')
