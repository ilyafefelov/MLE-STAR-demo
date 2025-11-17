#!/usr/bin/env python
"""
Orchestrator script to run ablation experiments across Gemini variants and datasets.

Usage:
  python scripts/run_experiments_variants.py --variants flash_lite,pro --datasets wine,iris --n-runs 5
"""
import argparse
import subprocess
import sys
from pathlib import Path

VARIANTS_TO_RUN = ['flash_lite','flash','pro']
DATASETS = ['breast_cancer','wine','digits','iris']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=str, default=','.join(VARIANTS_TO_RUN))
    parser.add_argument('--datasets', type=str, default=','.join(DATASETS))
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]

    for variant in variants:
        for dataset in datasets:
            outdir = Path(args.output_dir) / f"{dataset}_{variant}_n{args.n_runs}"
            outdir.mkdir(parents=True, exist_ok=True)
            cmd = [args.python, 'scripts/run_ablation.py', '--dataset', dataset, '--n-runs', str(args.n_runs), '--variant', variant, '--output-dir', str(outdir), '--verbose']
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd)

if __name__ == '__main__':
    main()
