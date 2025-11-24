#!/usr/bin/env python
"""Aggregate summary statistics from prior ablation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def find_summary_files(root: Path) -> List[Path]:
    pattern = "summary_statistics_"
    files = [p for p in root.rglob("*.csv") if pattern in p.name]
    files.sort()
    return files


def relative_experiment_name(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    # Remove the trailing summary file components: experiment/.../summary.csv -> experiment/sub
    parts = rel.parts[:-1]
    return "/".join(parts) if parts else rel.stem


def load_and_tag(file_path: Path, experiment: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.insert(0, "experiment", experiment)
    df.insert(1, "summary_file", str(file_path))
    return df


def compute_best(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for exp, g in df.groupby("experiment"):
        top = g.sort_values("mean", ascending=False).head(1).iloc[0]
        records.append(top)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ablation results into a single CSV")
    parser.add_argument(
        "--results-root",
        default="results",
        type=Path,
        help="Root directory containing experiment outputs",
    )
    parser.add_argument(
        "--output",
        default="results/aggregated_summary.csv",
        type=Path,
        help="Path to write the combined CSV",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Keep only the best configuration per experiment",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="do_print",
        help="Print the aggregated dataframe to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.results_root.resolve()
    if not root.exists():
        raise SystemExit(f"Results root not found: {root}")

    files = find_summary_files(root)
    if not files:
        raise SystemExit(f"No summary_statistics_*.csv files under {root}")

    frames = []
    for file_path in files:
        experiment = relative_experiment_name(root, file_path)
        frames.append(load_and_tag(file_path, experiment))

    combined = pd.concat(frames, ignore_index=True)
    if args.best_only:
        combined = compute_best(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    if args.do_print:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(combined)

    print(f"Wrote aggregated summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
