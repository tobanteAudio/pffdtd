#!/bin/sh

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/build/src/cpp/main_3d/pffdtd_3d"

sim_name="Modes"
sim_setup="${sim_name}_cpu.py"
sim_model_gen="${sim_name}_model.py"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"

model_dir="$root_dir/data/models/$sim_name"
materials_dir="$root_dir/data/materials"

fmin=20
fmax=400

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$python_dir"
python "$sim_model_gen"

# Generate sim data
cd "$python_dir"
python -m materials.build "$materials_dir"
python "$sim_setup"

# Run sim
cd "$sim_dir"
$engine_exe

# Post-process
cd "$python_dir"
python -m sim3d.process_outputs --data_dir="$sim_dir" --fcut_lowpass "$fmax" --N_order_lowpass=8 --symmetric --fcut_lowcut $fmin --N_order_lowcut=4 --air_abs_filter="none" --save_wav --plot
# python -m analysis.t60 --fmin=$fmin --fmax="$fmax" --target=0.25 ../../data/sim_data/$sim_name/cpu/R001_out_normalised.wav
# python -m analysis.t60 --data_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
python -m analysis.room_modes --data_dir="$sim_dir" --fmin=$fmin --fmax=$fmax --modes=20
