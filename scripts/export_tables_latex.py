#!/usr/bin/env python
"""
Export reports CSVs into LaTeX tables for inclusion in thesis.

Usage:
 python scripts/export_tables_latex.py --table1 reports/table1_model_comparison.csv --table2 reports/table2_ablation.csv --out reports
"""
import argparse
from pathlib import Path
import pandas as pd

def table1_to_latex(df, outpath: Path):
    # Keep relevant columns and order
    cols = ['dataset', 'variant', 'accuracy_mean', 'accuracy_std', 'generation_time', 'code_length', 'model_used']
    df = df[cols]
    df['accuracy_mean'] = df['accuracy_mean'].round(4)
    df['accuracy_std'] = df['accuracy_std'].round(4)
    latex = df.to_latex(index=False, longtable=True, caption='Model comparison: mean accuracy per dataset and Gemini variant', label='tab:gemini_model_comparison')
    outpath.write_text(latex, encoding='utf-8')
    print(f'Wrote LaTeX table to {outpath}')

def table2_to_latex(df, outpath: Path):
    # Select representative columns
    cols = ['dataset', 'variant', 'configuration', 'accuracy_mean', 'accuracy_std', 'n_runs']
    df = df[cols]
    df['accuracy_mean'] = df['accuracy_mean'].round(4)
    df['accuracy_std'] = df['accuracy_std'].round(4)
    latex = df.to_latex(index=False, longtable=True, caption='Ablation study: mean accuracy by configuration', label='tab:ablation')
    outpath.write_text(latex, encoding='utf-8')
    print(f'Wrote LaTeX table to {outpath}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table1', type=str, default='reports/table1_model_comparison.csv')
    parser.add_argument('--table2', type=str, default='reports/table2_ablation.csv')
    parser.add_argument('--out', type=str, default='reports')
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    t1 = Path(args.table1)
    t2 = Path(args.table2)

    if t1.exists():
        df1 = pd.read_csv(t1)
        table1_to_latex(df1, outdir / 'table1_model_comparison.tex')
    else:
        print('Table1 not found')

    if t2.exists():
        df2 = pd.read_csv(t2)
        table2_to_latex(df2, outdir / 'table2_ablation.tex')
    else:
        print('Table2 not found')

if __name__ == '__main__':
    main()
