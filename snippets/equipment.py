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


main()
