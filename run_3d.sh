#!/bin/sh

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
pffdtd_engine="$root_dir/build/src/cpp/pffdtd-engine"
# pffdtd_engine="$root_dir/cmake-build-cuda/src/cpp/pffdtd-engine"

sim_name="InfiniteBaffle"
sim_setup="${sim_name}.py"
sim_dir="$root_dir/sim_data/$sim_name/gpu"

model_dir="$root_dir/models/private/$sim_name"
model_dir="$root_dir/models/$sim_name"
materials_dir="$root_dir/materials"

fmin=20
fmax=1000
smoothing=0

# Delete old sim
rm -rf "$sim_dir"

# Generate materials, model & sim data
cd "$model_dir"
pffdtd sim3d setup "$sim_setup"

# Run sim
OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=16 $pffdtd_engine sim3d -e cuda -p "64" -s "$sim_dir"
# nsys profile --force-overwrite true -t cuda,osrt,nvtx --sample=none -o prof $pffdtd_engine sim3d -e cuda -p "64" -s "$sim_dir"
# /usr/local/cuda-12.6/bin/ncu \
#     --kernel-name-base demangled \
#     --kernel-name "regex:KernelAirCart" \
#     --launch-skip 20 --launch-count 1 \
#     --section="SpeedOfLight" --section="LaunchStats" --section="Occupancy" --section="MemoryWorkloadAnalysis" \
#     $pffdtd_engine sim3d -e cuda -p "64" -s "$sim_dir"

# pffdtd sim3d engine --sim_dir="$sim_dir" --plot --draw_backend="mayavi" --json_model="${model_dir}/model.json"

# Post-process
pffdtd sim3d process-outputs --sim_dir="$sim_dir" --resample_Fs 48000 --fcut_lowpass "$fmax" --order_lowpass=8 --symmetric_lowpass --fcut_lowcut "$fmin" --order_lowcut=4 --air_abs_filter="none" --save_wav --plot
pffdtd analysis summary --fmax="$fmax" --smoothing="$smoothing" $sim_dir/R001_out_normalised.wav
# pffdtd analysis response --musical --fmin="$fmin" --fmax="$fmax" --smoothing="$smoothing" $sim_dir/R001_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.39" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R002_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.06" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R003_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.25" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R004_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.55" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R005_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.96" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R006_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R001_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R015_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R030_out_normalised.wav
# pffdtd analysis rt60 --fmin=$fmin --fmax="$fmax" --target=0.296 $sim_dir/R001_out_normalised.wav
# pffdtd analysis rt60 --sim_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
# pffdtd analysis room-modes --sim_dir="$sim_dir" --fmin=$fmin --num_modes=20 --width=4.79 --length=7.13 --height=3.44
# pffdtd signals convolution --sim_dir="$sim_dir"
# pffdtd analysis localization --sim_dir="$sim_dir" "$model_dir/model.json"
# pffdtd diffusion measurement "$sim_dir"
