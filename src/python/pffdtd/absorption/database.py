# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import csv

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_BANDS = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]


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

    to_strip = [
        'description',
        'type',
        'trade name',
        'manufacturer',
        'surface',
        'layer thickness',
        'application',
        'dimensions',
        'distance',
        'weight/density',
        'flow resistance',
        'reference ',
    ]
    for scol in to_strip:
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

    df = df.rename(columns={
        'Unnamed: 43': 'character of absorption 2nd',
        'Unnamed: 45': 'material criteria 2nd',
        'Unnamed: 48': 'surface resistance 2nd',
    })

    for col in df.columns:
        if isinstance(col, str):
            if 'scattering' in col:
                df.drop(columns=col, inplace=True)

    df[125] = pd.to_numeric(df[125].replace(' - ', ''), errors='coerce').astype(float)
    df = df.dropna(axis=0, how='all', subset=ALL_BANDS)
    df[ALL_BANDS] = df[ALL_BANDS].replace(0.0, np.nan)

    df['manufacturer'] = df['manufacturer'].str.split('\n').str[0].str.strip()
    df['manufacturer'] = df['manufacturer'].str.split(';').str[0].str.strip()

    for col in ['character of absorption', 'material criteria', 'surface resistance']:
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)

        df[col + ' 2nd'] = df[col + ' 2nd'].fillna(0)
        df[col + ' 2nd'] = df[col + ' 2nd'].astype(int)

    character = pd.DataFrame.from_records([
        {'index': 0, 'character': ''},
        {'index': 1, 'character': 'low frequency absorbent'},
        {'index': 2, 'character': 'wide band absorbent'},
        {'index': 3, 'character': 'resonance absorbent'},
        {'index': 4, 'character': 'high frequency absorbent'},
        {'index': 5, 'character': 'poor absorbent'},
    ], index='index')
    df = pd.merge(
        df,
        character,
        how='inner',
        left_on='character of absorption',
        right_index=True,
    )
    df = pd.merge(
        df,
        character.rename(columns={'character': 'character 2nd'}),
        how='inner',
        left_on='character of absorption 2nd',
        right_index=True,
    )

    materials = pd.DataFrame.from_records([
        {'index': 0, 'material': ''},
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
    ], index='index')
    df = pd.merge(
        df,
        materials,
        how='inner',
        left_on='material criteria',
        right_index=True,
    )
    df = pd.merge(
        df,
        materials.rename(columns={'material': 'material 2nd'}),
        how='inner',
        left_on='material criteria 2nd',
        right_index=True,
    )

    material_resistance = pd.DataFrame.from_records([
        {'index': 0, 'material resistance': ''},
        {'index': 1, 'material resistance': 'suitable for gyms, (ball games)'},
        {'index': 2, 'material resistance': 'suitable for paint over'},
        {'index': 3, 'material resistance': 'waterproof, moisture resistant, washable'},
        {'index': 4, 'material resistance': 'heat resistant, fireproof (to various degrees)'},
    ], index='index')
    df = pd.merge(
        df,
        material_resistance,
        how='inner',
        left_on='surface resistance',
        right_index=True,
    )
    df = pd.merge(
        df,
        material_resistance.rename(columns={'material resistance': 'material resistance 2nd'}),
        how='inner',
        left_on='surface resistance 2nd',
        right_index=True,
    )

    df = df.drop(columns=[
        'character of absorption', 'material criteria', 'surface resistance',
        'character of absorption 2nd', 'material criteria 2nd', 'surface resistance 2nd',
    ])
    return df


@click.command(name='database')
@click.argument('database_excel', nargs=1, type=click.Path(exists=True))
@click.option('--plot', is_flag=True)
@click.option('--save_csv', type=click.Path())
def main(database_excel, plot, save_csv) -> None:
    df = read_absorption_database_excel(database_excel)
    print(df.columns)

    # df = df[df['manufacturer'].str.strip() !='']
    # df = df[df['material'] == 'stone, brick, concrete, clinker']
    # df = df[df['material'] == 'audiences']
    # df = df[df['description'].str.lower().str.contains('floor')]
    # df = df[df['description'].str.lower().str.contains('teppich')]
    # df = df[df['character'] != 'wide band absorbent']

    iso_octaves = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    col_keep = ALL_BANDS + [
        'description',
        'type',
        'trade name',
        'manufacturer',
        'layer thickness',
        'dimensions',
        'surface',
        'layer thickness',
        'application',
        'dimensions',
        'distance',
        'weight/density',
        'flow resistance',
        'character',
        'material',
        'material resistance',
        'character 2nd',
        'material 2nd',
        'material resistance 2nd',
        'reference',
    ]

    # df = df.dropna(subset=iso_octaves[1:-1], how='any')
    df = df.drop(columns=[col for col in df.columns if col not in col_keep])

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
