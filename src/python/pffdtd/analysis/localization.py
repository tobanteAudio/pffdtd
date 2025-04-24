# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import json
import pathlib
import sys

import h5py
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.signal import correlate

from pffdtd.common.wavfile import wavread


def _normalize(signal):
    return signal / np.max(np.abs(signal))


def _cross_correlation(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal1))
    return correlation, lags


def time_difference_of_arrival(signal1, signal2, fs):
    correlation, lags = _cross_correlation(signal1, signal2)
    lag_idx = np.argmax(correlation)  # Find the index of the maximum correlation
    tdoa = lags[lag_idx] / fs  # Convert lag index to time difference
    return tdoa


def tdoa_residuals(source_pos, mic_positions, tdoas, c):
    """Function to compute the residual between observed and estimated TDOAs.
    """
    estimated_tdoas = []
    for i, mic_i in enumerate(mic_positions):
        for j in range(i+1, len(mic_positions)):
            di = np.linalg.norm(source_pos - mic_i)
            dj = np.linalg.norm(source_pos - mic_positions[j])
            estimated_tdoas.append((di - dj) / c)
    return np.sum((np.array(estimated_tdoas) - tdoas)**2)


def tetrahedron_microphone_array(mic_positions, mic_sigs, fs, c=343.0, verbose=False):
    mic1 = _normalize(mic_sigs[0])
    mic2 = _normalize(mic_sigs[1])
    mic3 = _normalize(mic_sigs[2])
    mic4 = _normalize(mic_sigs[3])

    tdoa_12 = time_difference_of_arrival(mic1, mic2, fs)
    tdoa_13 = time_difference_of_arrival(mic1, mic3, fs)
    tdoa_14 = time_difference_of_arrival(mic1, mic4, fs)
    tdoa_23 = time_difference_of_arrival(mic2, mic3, fs)
    tdoa_24 = time_difference_of_arrival(mic2, mic4, fs)
    tdoa_34 = time_difference_of_arrival(mic3, mic4, fs)

    if verbose:
        print(f"TDOA between Mic1 and Mic2: {tdoa_12*1000:.4f} ms")
        print(f"TDOA between Mic1 and Mic3: {tdoa_13*1000:.4f} ms")
        print(f"TDOA between Mic1 and Mic4: {tdoa_14*1000:.4f} ms")
        print(f"TDOA between Mic2 and Mic3: {tdoa_23*1000:.4f} ms")
        print(f"TDOA between Mic2 and Mic4: {tdoa_24*1000:.4f} ms")
        print(f"TDOA between Mic3 and Mic4: {tdoa_34*1000:.4f} ms")

    tdoas = np.array([tdoa_12, tdoa_13, tdoa_14, tdoa_23, tdoa_24, tdoa_34])

    # initial_guess = np.array([1.25, 2.0, 1.6])
    # initial_guess = np.array([1, 1, 1])
    initial_guess = np.array([2.0, 2.0, 2.0])

    args = (mic_positions, tdoas, c)
    result = minimize(tdoa_residuals, initial_guess, args=args, tol=1e-10)
    # result = minimize(tdoa_residuals, initial_guess, args=args, tol=1e-10, bounds=[(-1, 4), (-1, 4), (-1, 4)])
    # result = differential_evolution(
    #     tdoa_residuals,
    #     bounds=[(-1, 4), (-1, 4), (-1, 4)],
    #     args=args,
    #     init='sobol',
    #     # popsize=1000,
    #     # x0=initial_guess,
    #     # tol=0.001,
    #     polish=False,
    #     # disp=True,
    # )

    # result = basinhopping(
    #     tdoa_residuals,
    #     x0=initial_guess,
    #     minimizer_kwargs={'args': args},
    #     # stepsize=0.001,
    #     niter=1000,
    #     # T=0.001,
    #     # disp=True,
    # )

    return result.x


def main():
    sim_dir = pathlib.Path(sys.argv[1])
    model_file = sys.argv[2]

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

    print(f'Distance 1-2 = {(distance_1-distance_2)/c*1000:.3f} ms')
    print(f'Distance 1-3 = {(distance_1-distance_3)/c*1000:.3f} ms')
    print(f'Distance 1-4 = {(distance_1-distance_4)/c*1000:.3f} ms')
    print(f'Distance 2-3 = {(distance_2-distance_3)/c*1000:.3f} ms')
    print(f'Distance 2-4 = {(distance_2-distance_4)/c*1000:.3f} ms')
    print(f'Distance 3-4 = {(distance_3-distance_4)/c*1000:.3f} ms')
    print('------------------------------')

    estimated_pos = tetrahedron_microphone_array(mic_pos, mic_sigs, fs, c=c, verbose=True)
    print('------------------------------')
    print(f'c:        {c} m/s')
    print(f'ACTUAL:   {source_pos}')
    print(f'ESTIMATE: {estimated_pos}')
    print(f'DISTANCE: {np.linalg.norm(source_pos-estimated_pos)}')


if __name__ == '__main__':
    main()
