# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import json
import pathlib

import click
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import correlate

from pffdtd.common.wavfile import wavread


def _tdoa_residuals(source_pos, mic_positions, tdoas, c):
    """Function to compute the residual between observed and estimated TDOAs.
    """
    estimated_tdoas = []
    for i, mic_i in enumerate(mic_positions):
        for j in range(i+1, len(mic_positions)):
            di = np.linalg.norm(source_pos - mic_i)
            dj = np.linalg.norm(source_pos - mic_positions[j])
            estimated_tdoas.append((di - dj) / c)
    return np.sum((np.array(estimated_tdoas) - tdoas)**2)


def time_difference_of_arrival(signal1, signal2, fs):
    correlation = correlate(signal1, signal2, mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal1))
    lag_idx = np.argmax(correlation)  # Find the index of the maximum correlation
    tdoa = lags[lag_idx] / fs         # Convert lag index to time difference
    return tdoa


def tetrahedron_microphone_array(mic_positions, mic_sigs, fs, c=343.0):
    norm = np.max(np.abs(mic_sigs))

    mic1 = mic_sigs[0]/norm
    mic2 = mic_sigs[1]/norm
    mic3 = mic_sigs[2]/norm
    mic4 = mic_sigs[3]/norm

    tdoas = np.array([
        time_difference_of_arrival(mic1, mic2, fs),
        time_difference_of_arrival(mic1, mic3, fs),
        time_difference_of_arrival(mic1, mic4, fs),
        time_difference_of_arrival(mic2, mic3, fs),
        time_difference_of_arrival(mic2, mic4, fs),
        time_difference_of_arrival(mic3, mic4, fs),
    ])

    args = (mic_positions, tdoas, c)
    initial_guess = np.array([2.0, 2.0, 2.0])
    result = minimize(_tdoa_residuals, initial_guess, args=args, tol=1e-10)
    return result.x, tdoas


@click.command(name='localization', help='Locate sound source.')
@click.argument('model_json', nargs=1, type=click.Path(exists=True))
@click.option('--sim_dir', type=click.Path(exists=True))
def main(model_json, sim_dir):
    sim_dir = pathlib.Path(sim_dir)
    model_file = model_json
    print(model_file)
    print(sim_dir)

    constants = h5py.File(sim_dir / 'constants.h5', 'r')
    c = float(constants['c'][...])

    fs1, mic1 = wavread(sim_dir/'R001_out_normalised.wav')
    fs2, mic2 = wavread(sim_dir/'R002_out_normalised.wav')
    fs3, mic3 = wavread(sim_dir/'R003_out_normalised.wav')
    fs4, mic4 = wavread(sim_dir/'R004_out_normalised.wav')
    assert fs1 == fs2
    assert fs1 == fs3
    assert fs1 == fs4

    fs = fs1
    mic_sigs = [mic1, mic2, mic3, mic4]

    with open(model_file, 'r') as f:
        model = json.load(f)

    source_pos = model['sources'][0]['xyz']
    mic_pos = np.array([
        model['receivers'][0]['xyz'],
        model['receivers'][1]['xyz'],
        model['receivers'][2]['xyz'],
        model['receivers'][3]['xyz'],
    ])

    distance_1 = np.linalg.norm(source_pos-mic_pos[0])
    distance_2 = np.linalg.norm(source_pos-mic_pos[1])
    distance_3 = np.linalg.norm(source_pos-mic_pos[2])
    distance_4 = np.linalg.norm(source_pos-mic_pos[3])
    actual_tdoas = [
        (distance_1-distance_2)/c*1000,
        (distance_1-distance_3)/c*1000,
        (distance_1-distance_4)/c*1000,
        (distance_2-distance_3)/c*1000,
        (distance_2-distance_4)/c*1000,
        (distance_3-distance_4)/c*1000,
    ]

    estimated_pos, estimated_tdoas = tetrahedron_microphone_array(mic_pos, mic_sigs, fs, c=c)
    estimated_tdoas *= 1000

    def report_error(a, b, actual, estimate):
        return {
            'A': a,
            'B': b,
            'Actual [ms]': actual,
            'Estimate [ms]': estimate,
            'Error [us]': (actual-estimate)*1000,
            'Rel-Error [%]': (estimate-actual)/actual*100,
        }

    errors = pd.DataFrame.from_records([
        report_error(1, 2, actual_tdoas[0], estimated_tdoas[0]),
        report_error(1, 3, actual_tdoas[1], estimated_tdoas[1]),
        report_error(1, 4, actual_tdoas[2], estimated_tdoas[2]),
        report_error(2, 3, actual_tdoas[3], estimated_tdoas[3]),
        report_error(2, 4, actual_tdoas[4], estimated_tdoas[4]),
        report_error(3, 4, actual_tdoas[5], estimated_tdoas[5]),
    ])

    print('------------------------------')
    print(f'c:        {c} m/s')
    print(f'ACTUAL:   {source_pos}')
    print(f'ESTIMATE: {estimated_pos}')
    print(f'DISTANCE: {np.linalg.norm(source_pos-estimated_pos)}')

    print('------------------------------')
    print(errors.to_markdown(index=False))
