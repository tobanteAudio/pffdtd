# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import sys
# import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv(sys.argv[1])
    df['ram/tdp'] = (df['ram']/df['tdp']).round(2)
    df['ram/price'] = (df['ram']/df['price']*1000).round(2)
    df['bandwidth/ram'] = (df['bandwidth']/df['ram']).round(2)
    df['bandwidth/tdp'] = (df['bandwidth']/df['tdp']).round(2)
    df['bandwidth/ram/tdp'] = df['bandwidth/ram']/df['tdp']
    # df['price/(bandwidth/ram)'] = df['price']/df['bandwidth/ram']
    # df['bandwidth/16GB'] = (df['bandwidth']/16.0).round(2)
    df = df.sort_values(by='ram/price', ascending=False)
    print(df.round(3).to_markdown(tablefmt='simple_grid', index=False))


main()
