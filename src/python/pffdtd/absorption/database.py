# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import csv

import click
import matplotlib.pyplot as plt
import pandas as pd


def read_absorption_database_excel(path) -> pd.DataFrame:
    """https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-16/ag-163/absorption-coefficient-database.html
    """
    df = pd.read_excel(
        path,
        sheet_name='selection_table',
        header=19,
        index_col=0,
        nrows=2574,
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

    for col in df.columns:
        if isinstance(col, str):
            if 'scattering' in col:
                df.drop(columns=col, inplace=True)

    bands = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    df[125] = pd.to_numeric(df[125].replace(' - ', ''), errors='coerce').astype(float)
    df = df.dropna(axis=0, how='all', subset=bands)

    df['description'] = df['description'].str[:200]

    df['manufacturer'] = df['manufacturer'].str.split('\n').str[0].str.strip()
    df['manufacturer'] = df['manufacturer'].str.split(';').str[0].str.strip()
    df['manufacturer'] = df['manufacturer'].str[:200]

    df['character of absorption'] = df['character of absorption'].fillna(0)
    df['character of absorption'] = df['character of absorption'].astype(int)

    df['material criteria'] = df['material criteria'].fillna(0)
    df['material criteria'] = df['material criteria'].astype(int)

    df = pd.merge(
        df,
        pd.DataFrame.from_records([
            {'index': 0, 'character': 'unkown'},
            {'index': 1, 'character': 'low frequency absorbent'},
            {'index': 2, 'character': 'wide band absorbent'},
            {'index': 3, 'character': 'resonance absorbent'},
            {'index': 4, 'character': 'high frequency absorbent'},
            {'index': 5, 'character': 'poor absorbent'},
        ], index='index'),
        how='inner',
        left_on='character of absorption',
        right_index=True,
    )

    df = pd.merge(
        df,
        pd.DataFrame.from_records([
            {'index': 0, 'material': 'unkown'},
            {'index': 1, 'material': 'mineral / rock wool / high absorbent'},
            {'index': 2, 'material': 'gypsum, plaster'},
            {'index': 3, 'material': 'wooden plates, chipboard'},
            {'index': 4, 'material': 'glass'},
            {'index': 5, 'material': 'metal'},
            {'index': 6, 'material': 'stone, brick, concrete, clinker'},
            {'index': 7, 'material': 'foam rubber  (hard / soft foam, polystyrol, polystyrene,) rubber'},
            {'index': 8, 'material': 'synthetic material, linoleum, hard plastics'},
            {'index': 9, 'material': 'tissues, carpets, textiles'},
            {'index': 10, 'material': 'paper, cardboard'},
            {'index': 11, 'material': 'audiences'},
            {'index': 12, 'material': 'miscellaneous'},
        ], index='index'),
        how='inner',
        left_on='material criteria',
        right_index=True,
    )

    df = df.drop(columns=['character of absorption', 'material criteria', 'type', 'trade name', 'reference '])
    return df


@click.command(name='database')
@click.argument('database_excel', nargs=1, type=click.Path(exists=True))
@click.option('--plot', is_flag=True)
@click.option('--save_csv', type=click.Path())
def main(database_excel, plot, save_csv) -> None:
    df = read_absorption_database_excel(database_excel)
    # df = df[df['manufacturer'] != '']
    # df = df[df['material'] == 'stone, brick, concrete, clinker']
    # df = df[df['material'] == 'audiences']
    # df = df[df['description'].str.lower().str.contains('floor')]
    # df = df[df['description'].str.lower().str.contains('teppich')]
    # df = df[df['character'] != 'wide band absorbent']

    iso_octaves = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    df = df.dropna(subset=iso_octaves[1:-1], how='any')
    df = df.drop(columns=[col for col in df.columns if col not in iso_octaves+['description', 'character', 'manufacturer']])

    if save_csv:
        df.to_csv(save_csv, sep=';', index=True, quoting=csv.QUOTE_MINIMAL)

    print(df)
    # print(df.to_string(max_colwidth=80, line_width=250, na_rep='', sparsify=True))
    print('--------------------------')

    bands = [{'freq': band, 'min': df[band].min(), 'max': df[band].max(), 'mean': df[band].mean()} for band in iso_octaves]
    mins = [band['min'] for band in bands]
    maxs = [band['max'] for band in bands]
    means = [band['mean'] for band in bands]

    if plot:
        for oct in iso_octaves:
            plt.scatter([oct]*len(df), df[oct])
        # plt.errorbar(iso_octaves, means, yerr=[mins, maxs], label='Glass')
        plt.semilogx(iso_octaves, means, label='Mean')
        # plt.semilogx(iso_octaves, maxs, label='Max')
        plt.xscale('log')
        plt.xlim(iso_octaves[0]/2, iso_octaves[-1]*2)
        plt.ylim(0.0, 1.0)
        plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        plt.legend()
        plt.show()
