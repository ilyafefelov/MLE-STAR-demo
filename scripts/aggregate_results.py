#!/usr/bin/env python
"""
Aggregate summary statistics from results/**/summary_statistics_*.csv into consolidated tables

Usage:
  python scripts/aggregate_results.py --results-dir results --outdir reports
"""
import argparse
import pandas as pd
from pathlib import Path
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--outdir', type=str, default='reports')
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_file in results_path.rglob('summary_statistics_*.csv'):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Could not read {csv_file}: {e}")
            continue

        # Try to infer dataset and variant from path
        parts = csv_file.parts
        # find index of 'results' and take next parts
        dataset_variant = 'unknown'
        try:
            i = parts.index('results')
            candidate = parts[i+1]
            # If the path uses 'variants_n*' folder, the dataset name is one level deeper
            if isinstance(candidate, str) and candidate.startswith('variants') and len(parts) > i+2:
                dataset_variant = parts[i+2]
            else:
                dataset_variant = candidate
        except ValueError:
            dataset_variant = 'unknown'

        # If dataset_variant is still a 'variants' placeholder or unknown, attempt to detect known dataset
        if not isinstance(dataset_variant, str) or dataset_variant.startswith('variants') or dataset_variant == 'unknown':
            KNOWN_DATASETS = ['breast_cancer', 'wine', 'digits', 'iris']
            found = False
            for p in parts:
                for ds in KNOWN_DATASETS:
                    if p == ds or p.startswith(ds + '_'):
                        dataset_variant = p
                        found = True
                        break
                if found:
                    break

        for _, r in df.iterrows():
            row = r.to_dict()
            row['dataset_variant'] = dataset_variant
            row['path'] = str(csv_file)
            rows.append(row)

    if not rows:
        print('No summary statistics found')
        return

    all_df = pd.DataFrame(rows)
    # Parse dataset and variant (if present)
    # Improve dataset/variant parsing: match known dataset names (allow underscores in dataset names)
    KNOWN_DATASETS = ['breast_cancer', 'wine', 'digits', 'iris']
    def split_dataset_variant(x):
        if not isinstance(x, str):
            return pd.Series({'dataset': x, 'variant': ''})
        for ds in KNOWN_DATASETS:
            if x == ds:
                return pd.Series({'dataset': ds, 'variant': ''})
            if x.startswith(ds + '_'):
                return pd.Series({'dataset': ds, 'variant': x[len(ds)+1:]})
        # Fallback: try last part after first underscore
        if '_' in x:
            parts = x.split('_', 1)
            return pd.Series({'dataset': parts[0], 'variant': parts[1]})
        return pd.Series({'dataset': x, 'variant': ''})

    dv = all_df['dataset_variant'].apply(split_dataset_variant)
    all_df = pd.concat([all_df, dv], axis=1)

    out_csv = outdir / 'aggregate_summary_stats.csv'
    all_df.to_csv(out_csv, index=False)
    print(f'Wrote aggregated summary to {out_csv}')

if __name__ == '__main__':
    main()
