#!/bin/sh

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch


set -e

build_dir=build
build_dir=cmake-build-acpp

root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/$build_dir/src/cpp/pffdtd-engine"

sim_name="Diffusor"
sim_dir="$root_dir/sim_data/$sim_name/cpu"
sim_setup="${sim_name}.py"
model_dir="$root_dir/models/$sim_name"

jobs=16

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$model_dir"
python "$sim_setup"

# Run sim
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=$jobs OMP_NUM_THREADS=$jobs "$engine_exe" sim2d -p 32 -s "$sim_dir" -e sycl
# pffdtd sim2d run --sim_dir "$sim_dir" --video

# Post-process
pffdtd sim2d process-outputs --fmin=20 --sim_dir="$sim_dir" "$sim_dir/out.h5"

# Report
# pffdtd sim2d report --sim_dir="$sim_dir" "$sim_dir/out.h5"
pffdtd diffusion measurement "$sim_dir"
