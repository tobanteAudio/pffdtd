
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import pathlib
import sys

import h5py
import numpy as np

sim_dir = pathlib.Path(sys.argv[1])

with h5py.File(sim_dir / 'out-mojo.h5', 'r') as f:
    out_cpp = f['out'][...]

with h5py.File(sim_dir / 'out.h5', 'r') as f:
    out_mojo = f['out'][...]

print(np.max(np.abs(out_mojo - out_cpp)))

assert not np.isnan(out_mojo).any()
assert not np.isinf(out_mojo).any()
assert not np.isnan(out_cpp).any()
assert not np.isinf(out_cpp).any()
assert np.allclose(out_mojo, out_cpp, atol=1e-6)
