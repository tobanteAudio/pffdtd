#!/bin/sh

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
pffdtd_engine="$root_dir/build/src/cpp/pffdtd-engine"
# pffdtd_engine="$root_dir/cmake-build-cuda/src/cpp/pffdtd-engine"

sim_name="BassReflex"
sim_setup="${sim_name}.py"
sim_dir="$root_dir/sim_data/$sim_name/gpu"

model_dir="$root_dir/models/$sim_name"
materials_dir="$root_dir/materials"

fmin=20
fmax=6000
smoothing=0

# Delete old sim
rm -rf "$sim_dir"

# Generate materials, model & sim data
cd "$model_dir"
pffdtd sim3d setup "$sim_setup"

# Run sim
OMP_PROC_BIND=spread OMP_NUM_THREADS=16 $pffdtd_engine sim3d -e cuda -p "64" -s "$sim_dir"
# pffdtd sim3d engine --sim_dir="$sim_dir" --plot --draw_backend="mayavi" --json_model="${model_dir}/model.json"

# Post-process
pffdtd sim3d process-outputs --sim_dir="$sim_dir" --fcut_lowpass "$fmax" --order_lowpass=8 --symmetric_lowpass --fcut_lowcut "$fmin" --order_lowcut=4 --air_abs_filter="none" --save_wav --plot
pffdtd analysis summary --fmax="$fmax" --smoothing="$smoothing" $sim_dir/R001_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.51" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R002_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.13" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R003_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.31" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R004_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-7.62" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R005_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-8.02" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R006_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R001_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R015_out_normalised.wav
# pffdtd analysis spectrogram --min_db=-60 $sim_dir/R030_out_normalised.wav
# pffdtd analysis rt60 --fmin=$fmin --fmax="$fmax" --target=0.3 $sim_dir/R001_out_normalised.wav
# pffdtd analysis rt60 --sim_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
# pffdtd analysis room-modes --sim_dir="$sim_dir" --fmin=$fmin --num_modes=20 --width=3.65 --length=6.0 --height=3.12
# pffdtd signals convolution --sim_dir="$sim_dir"
# pffdtd analysis localization --sim_dir="$sim_dir" "$model_dir/model.json"
# pffdtd diffusor measurement "$sim_dir"
