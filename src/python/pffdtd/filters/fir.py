# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch


def main():
    fs = 96000
    taps = 3072*2+1
    taps = 1024
    taps = fs/1000*20
    resolution = fs/taps
    group_delay = (taps-1)/2
    print(f'{fs=}Hz')
    print(f'{taps=}')
    print(f'{resolution=:.2f}Hz')
    print(f'{group_delay=} samples')
    print(f'group_delay={1000/fs*group_delay:.2f} ms')


if __name__ == '__main__':
    main()
