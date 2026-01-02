# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

def main():
    fs = 96000
    bitrate = 32
    sweep_length = 10
    tail_length = 3

    speakers = 7
    positions = 16

    recordings = speakers*positions
    recording_length = (sweep_length*tail_length*recordings)
    recording_storage = fs*(bitrate/8)*recording_length

    print(f'Recordings   = {recordings}')
    print(f'Total length = {recording_length:.0f} s / {recording_length/60:.0f} min')
    print(f'Storage      = {recording_storage/1e6:.3f} MB')


main()
