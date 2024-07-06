import glob
import os
import sys

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt


def collect_wav_files(directory, pattern="*.wav"):
    search_pattern = os.path.join(directory, pattern)
    wav_files = glob.glob(search_pattern)
    return wav_files


def octave_filter(sig, fs, center_freq):
    # Design octave bandpass filter
    low_freq = center_freq / np.sqrt(2)
    high_freq = center_freq * np.sqrt(2)
    sos = signal.butter(4, [low_freq, high_freq],
                        btype='band', fs=fs, output='sos')
    filtered_signal = signal.sosfilt(sos, sig)
    return filtered_signal


def third_octave_filter(sig, fs, center):
    factor = 2 ** (1/6)  # One-third octave factor
    low = center / factor
    high = center * factor
    sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, sig)


def energy_decay_curve(ir):
    edc = np.cumsum(ir[::-1]**2)[::-1]
    edc_db = 10 * np.log10(edc / np.max(edc))
    return edc_db


def calculate_t60(edc_db, fs):
    t = np.arange(len(edc_db)) / fs
    edc_db -= np.max(edc_db)  # Normalize to 0 dB at the start
    start_idx = np.where(edc_db <= -5)[0][0]
    end_idx = np.where(edc_db <= -35)[0][0]
    t60 = 2 * (t[end_idx] - t[start_idx])
    return t60


def ebu_3000_t60_threshold_upper(freqs):
    times = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        if freq < 63:
            times[i] = 0.3
        elif freq <= 200:
            times[i] = 0.3 - (0.25 * (freq - 63) / (200 - 63))
        else:
            times[i] = 0.05
    return times


def ebu_3000_t60_threshold_lower(freqs):
    times = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        if freq < 4000:
            times[i] = -0.05
        else:
            times[i] = -0.10
    return times


def main():
    center_freqs = [15.625, 31.5, 63, 125, 250, 500, 1000, 2000]
    center_freqs = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000
    ]
    # 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000

    directory = sys.argv[1]
    files = collect_wav_files(directory, "*_out_normalised.wav")

    file_times = []
    for file in files:
        fs, ir = wavfile.read(file)
        t60_times = []
        print(f"{file}")
        for center_freq in center_freqs:
            filtered_ir = third_octave_filter(ir, fs, center_freq)
            edc_db = energy_decay_curve(filtered_ir)
            t60 = calculate_t60(edc_db, fs)
            t60_times.append(round(t60, 3))
            print(f"T60 at {center_freq} Hz: {t60:.2f} seconds")

        file_times.append(np.array(t60_times))

    fig = plt.figure()

    # T60
    ax = fig.add_subplot(2, 1, 1)
    ax.margins(0, 0.1)
    ax.semilogx(center_freqs, file_times[0])

    ax.set_title("T60")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Decay [s]")

    ax.set_xlim((center_freqs[0], center_freqs[-1]))
    ax.set_ylim((0.0, np.max(file_times[0])+0.1))
    ax.grid(which='both', axis='both')

    # Tolerance
    diff = np.insert(np.diff(file_times[0]), 0, 0.0)
    diff = np.insert(file_times[0][:-1]-file_times[0][1:], 0, 0.0)

    ax = fig.add_subplot(2, 1, 2)
    ax.margins(0, 0.1)
    ax.semilogx(center_freqs, diff)
    ax.semilogx(center_freqs, ebu_3000_t60_threshold_upper(center_freqs))
    ax.semilogx(center_freqs, ebu_3000_t60_threshold_lower(center_freqs))

    ax.set_title(f"T60 Tolerance (EBU Tech 3000)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Difference [s]")

    ax.set_xlim((center_freqs[0], center_freqs[-1]))
    ax.set_ylim((np.min(diff)-0.1, 0.4))
    ax.grid(which='both', axis='both')

    plt.show()


main()
