#!/usr/bin/env python
"""
Aggregate model comparison JSON files into a consolidated CSV

Usage:
  python scripts/aggregate_model_comparison.py --input-dir model_comparison_results --out reports/table1_model_comparison.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='model_comparison_results')
    parser.add_argument('--out', type=str, default='reports/table1_model_comparison.csv')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows_by_key = {}
    for json_file in input_dir.glob('comparison_full_*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not load {json_file}: {e}")
            continue

        for record in data:
            # Only consider successful records with metrics
            if not record.get('success', True):
                continue
            key = (record.get('dataset'), record.get('model'))
            if key not in rows_by_key:
                rows_by_key[key] = []
            rows_by_key[key].append(record)

    if not rows_by_key:
        print('No model comparison JSON files found')
        return

    # Reduce aggregated records per dataset+variant
    reduced_rows = []
    for (dataset, model), recs in rows_by_key.items():
        # Average numeric metrics where present
        nums = ['accuracy_mean', 'accuracy_std', 'generation_time', 'code_length']
        out = {'dataset': dataset, 'variant': model}
        for n in nums:
            vals = [r.get(n) for r in recs if r.get(n) is not None]
            out[n] = sum(vals) / len(vals) if vals else None
        # model_used: pick most common or first non-empty
        model_used_vals = [r.get('model_used') for r in recs if r.get('model_used')]
        out['model_used'] = model_used_vals[0] if model_used_vals else None
        pipeline_steps_vals = [r.get('pipeline_steps') for r in recs if r.get('pipeline_steps')]
        if pipeline_steps_vals and isinstance(pipeline_steps_vals[0], list):
            out['pipeline_steps'] = '|'.join(pipeline_steps_vals[0])
        else:
            out['pipeline_steps'] = pipeline_steps_vals[0] if pipeline_steps_vals else ''
        reduced_rows.append(out)

    df = pd.DataFrame(reduced_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote model comparison summary to {out_path}')

if __name__ == '__main__':
    main()
