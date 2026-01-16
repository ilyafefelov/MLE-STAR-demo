#!/usr/bin/env python
"""
Clean and deduplicate generated reports (table1_model_comparison.csv, aggregate_summary_stats.csv, table2_ablation.csv)

Usage:
 python scripts/clean_reports.py --reports reports
"""
import argparse
from pathlib import Path
import pandas as pd

def clean_table1(path: Path):
    df = pd.read_csv(path)
    # Drop approx empty records where accuracy_mean is NaN and other metrics are missing
    df_clean = df[~df['accuracy_mean'].isna()]
    # Deduplicate: keep the last occurrence per dataset+variant
    df_clean = df_clean.sort_values(by=['dataset', 'variant']).drop_duplicates(subset=['dataset', 'variant'], keep='last')
    df_clean.to_csv(path, index=False)
    print(f'Cleaned table1: {path}')

def clean_aggregate(path: Path):
    df = pd.read_csv(path)
    # Fix dataset/variant for known datasets
    KNOWN_DATASETS = ['breast_cancer', 'wine', 'digits', 'iris']
    def split_dataset_variant(x):
        if not isinstance(x, str):
            return None, ''
        for ds in KNOWN_DATASETS:
            if x == ds:
                return ds, ''
            if x.startswith(ds + '_'):
                return ds, x[len(ds)+1:]
        if '_' in x:
            parts = x.split('_', 1)
            return parts[0], parts[1]
        return x, ''

    ds, variant = zip(*[split_dataset_variant(x) for x in df['dataset_variant']])
    df['dataset'] = ds
    df['variant'] = variant
    # Save back
    df.to_csv(path, index=False)
    print(f'Cleaned aggregate: {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=str, default='reports')
    args = parser.parse_args()
    outdir = Path(args.reports)
    t1 = outdir / 'table1_model_comparison.csv'
    agg = outdir / 'aggregate_summary_stats.csv'
    t2 = outdir / 'table2_ablation.csv'
    if t1.exists():
        clean_table1(t1)
    if agg.exists():
        clean_aggregate(agg)
    # Rebuild Table2
    if agg.exists():
        df = pd.read_csv(agg)
        cols = ['dataset', 'variant', 'configuration', 'mean', 'std', 'n_runs', 'ci_lower', 'ci_upper']
        df_short = df[cols].copy()
        df_short.rename(columns={'mean':'accuracy_mean', 'std':'accuracy_std'}, inplace=True)
        df_short.to_csv(t2, index=False)
        print(f'Rebuilt table2: {t2}')
    # Deduplicate table2 entries
    if t2.exists():
        df_t2 = pd.read_csv(t2)
        before = len(df_t2)
        df_t2 = df_t2.drop_duplicates(subset=['dataset', 'variant', 'configuration', 'n_runs'], keep='last')
        df_t2.to_csv(t2, index=False)
        print(f'Deduplicated table2: {t2} ({before} -> {len(df_t2)} rows)')
