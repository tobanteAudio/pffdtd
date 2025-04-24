# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
"""https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-16/ag-163/absorption-coefficient-database.html
"""

import csv
import sys

import pandas as pd


def load_database(path):
    def column_filter(col):
        if isinstance(col, str):
            col = col.strip()
            if col in ['No.', 'description', 'type', 'trade name', 'layer thickness', 'flow resistance', 'character of absorption', 'reference']:
                return True

        try:
            _ = int(col)
            return True
        except:
            return False

    df = pd.read_excel(
        path,
        sheet_name='selection_table',
        header=19,
        index_col=0,
        nrows=2574,
        usecols=column_filter
    )

    for scol in ['description', 'type', 'trade name', 'flow resistance', 'reference ']:
        df[scol.strip()] = (
            df[scol].str.strip()
            .str.split(' ')
            .str.join(' ')
            .str.split('\n')
            .str.join(' ')
            .str.split('\r')
            .str.join(' ')
            .str.split('\t')
            .str.join(' ')
            .str.split('\r\n')
            .str.join(' ')
        )

    df = df.drop(['type', 'trade name', 'reference '], axis=1)
    df = df.dropna(axis=0, how='all', subset=[63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
    df = df.dropna(axis=0, subset=['character of absorption'])
    df['character of absorption'] = df['character of absorption'].astype(int)
    # df['description'] = df['description']
    # df['reference'] = df['reference']

    df = pd.merge(
        df,
        pd.DataFrame.from_records([
            {'index': 1, 'type': 'low frequency absorbent'},
            {'index': 2, 'type': 'wide band absorbent'},
            {'index': 3, 'type': 'resonance absorbent'},
            {'index': 4, 'type': 'high frequency absorbent'},
            {'index': 5, 'type': 'poor absorbent'},
        ], index='index'),
        how='inner',
        left_on='character of absorption',
        right_index=True,
    )

    df = df.drop(['character of absorption', 'flow resistance', 'layer thickness'], axis=1)
    return df.sort_values(['reference', 63, 80, 100, 125, 160, 200], ascending=False)


def main():
    df: pd.DataFrame = load_database(sys.argv[1])
    df.to_csv('absorber.csv', sep=';', index=False, quoting=csv.QUOTE_MINIMAL)
    print(df)
    # print(df.columns)
    print('--------------------------')
    # print(df.describe())
    # print(df.memory_usage(deep=True))


if __name__ == '__main__':
    main()
