#!/usr/bin/env python
"""
Build ablation table from aggregated summary stats file into a human-friendly CSV

Usage:
 python scripts/build_table2_ablation.py --agg reports/aggregate_summary_stats.csv --out reports/table2_ablation.csv
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', type=str, default='reports/aggregate_summary_stats.csv')
    parser.add_argument('--out', type=str, default='reports/table2_ablation.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.agg)
    # Keep only needed columns
    df_short = df[['dataset', 'variant', 'configuration', 'mean', 'std', 'n_runs', 'ci_lower', 'ci_upper']].copy()
    df_short.rename(columns={'mean':'accuracy_mean', 'std':'accuracy_std'}, inplace=True)
    df_short.to_csv(args.out, index=False)
    print(f'Wrote Table2 ablation to {args.out}')

if __name__ == '__main__':
    main()
