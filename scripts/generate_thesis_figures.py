#!/usr/bin/env python
"""
Generate thesis-ready figures from aggregated reports

Usage:
 python scripts/generate_thesis_figures.py --table1 reports/table1_model_comparison.csv --table2 reports/table2_ablation.csv --out reports/figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def make_variant_barplot(df_table1, outpath: Path):
    # Plot mean accuracy per dataset grouped by variant
    df = df_table1.copy()
    # Keep only necessary columns
    df = df[['dataset', 'variant', 'accuracy_mean']]
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='dataset', y='accuracy_mean', hue='variant')
    plt.ylim(0, 1.05)
    plt.title('Model comparison: Accuracy by dataset and Gemini variant')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    outpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath / 'model_comparison_barplot.png')
    plt.close()
    print('Saved model comparison barplot')

def make_ablation_plots(df_table2, outpath: Path):
    outpath.mkdir(parents=True, exist_ok=True)
    # For each dataset, create a grouped bar plot of mean accuracy by configuration and variant (if multiple variants present)
    for ds, grp_ds in df_table2.groupby('dataset'):
        plt.figure(figsize=(12,6))
        # pivot by variant and configuration
        pivot = grp_ds.pivot_table(index='configuration', columns='variant', values='accuracy_mean')
        pivot_err = grp_ds.pivot_table(index='configuration', columns='variant', values='accuracy_std')
        # plot manually for proper yerr shape
        n_variants = len(pivot.columns)
        n_configs = len(pivot.index)
        width = 0.8 / n_variants
        x = np.arange(n_configs)
        for i, variant in enumerate(pivot.columns):
            heights = pivot[variant].values
            errs = pivot_err[variant].values if (variant in pivot_err.columns) else np.zeros_like(heights)
            xpos = x - 0.4 + i * width + width/2
            plt.bar(xpos, heights, width=width, yerr=errs, capsize=4, label=variant)
        plt.ylim(0,1.05)
        plt.xticks(x, pivot.index, rotation=45, ha='right')
        plt.title(f'Ablation: {ds} — Accuracy by configuration and variant')
        plt.ylabel('Accuracy')
        plt.xlabel('Configuration')
        plt.legend()
        plt.tight_layout()
        filename = outpath / f'ablation_{ds}.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved ablation figure for {ds}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table1', type=str, default='reports/table1_model_comparison.csv')
    parser.add_argument('--table2', type=str, default='reports/table2_ablation.csv')
    parser.add_argument('--out', type=str, default='reports/figures')
    args = parser.parse_args()

    t1 = Path(args.table1)
    t2 = Path(args.table2)
    outdir = Path(args.out)
    if t1.exists():
        df1 = pd.read_csv(t1)
        make_variant_barplot(df1, outdir)
    else:
        print('Table1 not found, skipping model comparison barplot')

    if t2.exists():
        df2 = pd.read_csv(t2)
        make_ablation_plots(df2, outdir)
    else:
        print('Table2 not found, skipping ablation plots')

if __name__ == '__main__':
    main()
