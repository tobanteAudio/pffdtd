# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import pathlib
import sys

import h5py
import numpy as np


def main():
    path_a = sys.argv[1]
    path_b = sys.argv[2]

    with h5py.File(path_a, 'r') as f:
        out_a = f['out'][...]
    with h5py.File(path_b, 'r') as f:
        out_b = f['out'][...]

    assert not np.isnan(out_b).any()
    assert not np.isinf(out_b).any()
    assert not np.isnan(out_a).any()
    assert not np.isinf(out_a).any()

    max_error = np.max(np.abs(out_b - out_a))
    print(f'Max-Error: {max_error:.5e}')
    print(f'Max-Error: {20*np.log10(max_error):.4f} dB')


if __name__ == '__main__':
    main()
