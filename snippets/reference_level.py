# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import pandas as pd
import numpy as np


def main():
    ref_level = 79
    crest_factor = 6

    models = pd.DataFrame.from_records([
        {'model': '8341A', 'short-term': 110, 'long-term': 101},
        {'model': '8351B', 'short-term': 113, 'long-term': 103},
        {'model': '8361A', 'short-term': 118, 'long-term': 109},
    ])

    speakers = pd.DataFrame.from_records([
        {'position': 'Center', 'distance': 3867, 'model': '8361A'},
        {'position': 'Left', 'distance': 4303, 'model': '8361A'},
        {'position': 'Wide', 'distance': 3479, 'model': '8351B'},
        {'position': 'Surround', 'distance': 2850, 'model': '8351B'},
        {'position': 'Rear', 'distance': 4735, 'model': '8351B'},
        {'position': 'Top-Front', 'distance': 2955, 'model': '8341A'},
        {'position': 'Top-Surround', 'distance': 1900, 'model': '8341A'},
        {'position': 'Top-Rear', 'distance': 3312, 'model': '8341A'},
    ])

    speakers = pd.merge(speakers, models, on='model')
    speakers['distance'] = speakers['distance']/1000
    speakers['SPL @ 1m'] = ref_level + 20*np.log10(speakers['distance'])
    speakers['headroom long-term'] = speakers['long-term']-speakers['SPL @ 1m']
    speakers['headroom short-term'] = speakers['short-term']-speakers['SPL @ 1m']-crest_factor
    print(speakers.round(2).to_markdown(tablefmt='simple_grid', index=False))


if __name__ == '__main__':
    main()
