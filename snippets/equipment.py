# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import sys
import pandas as pd


def main():
    df = pd.read_csv(sys.argv[1])
    df['count'] = df['count'].fillna(0).astype(int)
    df['cost'] = df['price']*df['count']
    df = df[df['cost'] > 0]
    df = df.sort_values('cost', ascending=False)
    total = df['cost'].sum()
    total_weight = (df['weight']*df['count']).sum()

    print(df.to_markdown(tablefmt='simple_grid', index=False))

    for group, gdf in df.groupby('type'):
        print(f'{group:<10} = €{gdf["cost"].sum():<5} / {(gdf["weight"]*gdf["count"]).sum():.1f} kg')
        # print(gdf.to_markdown(tablefmt="simple_grid", index=False))
    print(f'{"TOTAL":<10} = €{total:<5} / {total_weight:.1f} kg')

    power = {
        'Focusrite Red 8Line': 120/2,
        'RME 12Mic-D': 20,
        'Tascam 4IN': 8.5,
        'Tascam 4OUT': 7.5,
        'Tascam AES': 3,
        'Mac Mini': 65/2,
        '27inch HD Monitor': 30,
        'Switch': 10*2,
        'Environment Sensor': 5,
        'Motorized Stands': 10,
        'Genelec 8341': 55/2,
        # "Genelec 7360A": 30,
    }

    total_power = sum(v for k, v in power.items())
    print('---------------')
    print(f'Power:    {total_power:.0f} W')
    print(f'Duration: {4000/total_power:.1f} hours')


main()
