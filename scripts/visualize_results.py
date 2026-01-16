#!/usr/bin/env python
"""Visualize aggregated metrics from automated evaluation.

Auto-discovers latest `auto_evaluation_*.csv` in `model_comparison_results/` if
`--csv` not provided. Aggregates mean metric_value per (task, artifact_type) and
outputs a bar chart PNG plus a JSON summary.

Usage:
  python scripts/visualize_results.py
  python scripts/visualize_results.py --csv model_comparison_results/auto_evaluation_20250101_120000.csv --show
  python scripts/visualize_results.py --style seaborn-v0_8

Outputs:
  model_comparison_results/results_summary_<timestamp>.png
  model_comparison_results/results_summary_<timestamp>.json

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def discover_latest_csv(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("auto_evaluation_*.csv"))
    if not files:
        raise FileNotFoundError(f"No auto_evaluation_*.csv files in {results_dir}")
    return files[-1]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    required = {"task", "artifact_type", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    grouped = (df.groupby(["task", "artifact_type"]) ["metric_value"].agg(['mean','count']).reset_index())
    grouped.rename(columns={'mean':'mean_metric','count':'count'}, inplace=True)
    return grouped


def to_nested(grouped: pd.DataFrame) -> Dict:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, row in grouped.iterrows():
        task = row['task']
        art = row['artifact_type']
        mean_val = float(row['mean_metric'])
        cnt = int(row['count'])
        summary.setdefault(task, {})[art] = {"mean": mean_val, "count": cnt}
    return summary


def plot(grouped: pd.DataFrame, style: str, out_png: Path):
    try:
        plt.style.use(style)
    except Exception:
        print(f"[warn] Style '{style}' unavailable; using default.")
    tasks = list(grouped['task'].unique())
    arts = list(grouped['artifact_type'].unique())
    n_tasks = len(tasks)
    n_arts = len(arts)
    bar_width = 0.8 / max(n_arts,1)
    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 1.3), 4 + n_arts))
    for i, art in enumerate(arts):
        subset = grouped[grouped['artifact_type'] == art]
        xs = []
        ys = []
        for t_i, t in enumerate(tasks):
            match = subset[subset['task'] == t]
            ys.append(float(match['mean_metric'].iloc[0]) if not match.empty else 0.0)
            xs.append(t_i + i * bar_width)
        ax.bar(xs, ys, width=bar_width, label=art)
    centers = [i + (n_arts - 1) * bar_width / 2 for i in range(n_tasks)]
    ax.set_xticks(centers)
    ax.set_xticklabels(tasks, rotation=18, ha='right')
    ax.set_ylabel('Mean metric_value')
    ax.set_title('Mean metric_value per (task, artifact_type)')
    if n_arts > 1:
        ax.legend(title='artifact_type', fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    return fig


def maybe_show(fig, show: bool):
    if not show:
        return
    backend = matplotlib.get_backend().lower()
    if backend.startswith('agg'):
        print('[info] Headless backend; skipping interactive display.')
        return
    try:
        fig.show()
    except Exception as e:
        print(f"[warn] Could not display figure: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics.')
    parser.add_argument('--csv', type=str, help='Explicit CSV path. If omitted auto-discovers latest.')
    parser.add_argument('--output-dir', type=str, default='model_comparison_results', help='Directory with evaluation CSVs.')
    parser.add_argument('--style', type=str, default='seaborn-v0_8', help='Matplotlib style name.')
    parser.add_argument('--show', action='store_true', help='Attempt to show interactive window.')
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    if not results_dir.exists():
        print(f"[error] Results dir not found: {results_dir}")
        sys.exit(1)

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            print(f"[error] CSV not found: {csv_path}")
            sys.exit(1)
    else:
        try:
            csv_path = discover_latest_csv(results_dir)
        except Exception as e:
            print(f"[error] {e}")
            sys.exit(1)

    print(f"[info] Using CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[error] Failed to read CSV: {e}")
        sys.exit(1)

    try:
        grouped = aggregate(df)
    except Exception as e:
        print(f"[error] Aggregation failed: {e}")
        sys.exit(1)

    summary = to_nested(grouped)
    ts = time.strftime('%Y%m%d_%H%M%S')
    png_path = results_dir / f'results_summary_{ts}.png'
    json_path = results_dir / f'results_summary_{ts}.json'
    fig = plot(grouped, args.style, png_path)

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[error] Failed writing JSON: {e}")
        sys.exit(1)

    print(f"[info] Wrote PNG: {png_path}")
    print(f"[info] Wrote JSON: {json_path}")
    maybe_show(fig, args.show)


if __name__ == '__main__':
    main()
