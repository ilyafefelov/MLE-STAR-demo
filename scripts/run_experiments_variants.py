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
    parser.add_argument('--generate-live', action='store_true', help='Generate pipelines live via MLE-STAR/Gemini before running ablation')
    parser.add_argument('--env-file', type=str, default='.env', help='Env file to load (for GEMINI_API_TOKEN) when generating live pipelines')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]

    for variant in variants:
        for dataset in datasets:
            outdir = Path(args.output_dir) / f"{dataset}_{variant}_n{args.n_runs}"
            outdir.mkdir(parents=True, exist_ok=True)

            # Optionally generate pipeline live via MLE-STAR
            pipeline_file = None
            if args.generate_live:
                variant_model_map = {
                    'flash_lite': 'gemini-2.5-flash-lite',
                    'flash': 'gemini-2.5-flash',
                    'pro': 'gemini-2.5-pro'
                }
                model = variant_model_map.get(variant)
                if model:
                    gen_cmd = [args.python, 'scripts/generate_pipeline.py', '--dataset', dataset, '--out', f'model_comparison_results/gemini_live_{variant}_{dataset}.py', '--env-file', args.env_file, '--model', model]
                    print('Generating live pipeline:', ' '.join(gen_cmd))
                    subprocess.run(gen_cmd)
                    # Use the generated file as pipeline-file
                    pipeline_file = f'model_comparison_results/gemini_live_{variant}_{dataset}.py'

            cmd = [args.python, 'scripts/run_ablation.py', '--dataset', dataset, '--n-runs', str(args.n_runs), '--variant', variant, '--output-dir', str(outdir), '--verbose']
            if args.generate_live and pipeline_file:
                cmd.extend(['--pipeline-file', pipeline_file])
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd)

if __name__ == '__main__':
    main()
