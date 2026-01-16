#!/usr/bin/env python
"""
Orchestrate pipeline generation via MLE-STAR/Gemini and run ablation experiments on the generated pipelines.

Usage:
 python scripts/run_live_generation_and_ablation.py --datasets breast_cancer,wine,digits,iris --n-runs 20
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys

def load_env(env_file: Path):
    if not env_file.exists():
        return
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k] = v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='breast_cancer,wine,digits,iris')
    parser.add_argument('--n-runs', type=int, default=20)
    parser.add_argument('--env-file', type=str, default='.env')
    parser.add_argument('--agent-script', type=str, default='scripts/run_mle_star_programmatic.py')
    parser.add_argument('--model', type=str, default=None, help='Root agent model to use for generation (e.g. gemini-2.5-flash, gemini-2.5-pro)')
    parser.add_argument('--python', type=str, default=sys.executable)
    args = parser.parse_args()

    load_env(Path(args.env_file))
    # Map GEMINI_API_TOKEN to GOOGLE_API_KEY
    if 'GEMINI_API_TOKEN' in os.environ and 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']

    datasets = [d.strip() for d in args.datasets.split(',')]
    for ds in datasets:
        print(f'=== Generating pipeline for: {ds} ===')
        # call generate_pipeline.py to create a live Perl pipeline file
        out_file = f'model_comparison_results/gemini_live_{ds}.py'
        gen_cmd = [args.python, 'scripts/generate_pipeline.py', '--dataset', ds, '--out', out_file, '--env-file', args.env_file, '--agent-script', args.agent_script]
        if args.model:
            gen_cmd.extend(['--model', args.model])
        proc = subprocess.run(gen_cmd)
        if proc.returncode != 0:
            print(f'Generation for {ds} failed; check logs and credentials.')
            continue
        # Now run ablation using the generated pipeline file
        print(f'=== Running ablation for: {ds} with {args.n_runs} runs ===')
        out_dir = Path('results') / f'live_{ds}_n{args.n_runs}'
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [args.python, 'scripts/run_ablation.py', '--dataset', ds, '--n-runs', str(args.n_runs), '--pipeline-file', out_file, '--output-dir', str(out_dir), '--verbose']
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f'Ablation run for {ds} failed (return code {proc.returncode}); check logs.')
            continue

    print('All generation and ablation runs finished — aggregating results')
    subprocess.run([args.python, 'scripts/aggregate_results.py', '--results-dir', 'results', '--outdir', 'reports'])
    subprocess.run([args.python, 'scripts/clean_reports.py', '--reports', 'reports'])
    subprocess.run([args.python, 'scripts/export_tables_latex.py', '--table1', 'reports/table1_model_comparison.csv', '--table2', 'reports/table2_ablation.csv', '--out', 'reports'])
    subprocess.run([args.python, 'scripts/generate_report_figures.py', '--table1', 'reports/table1_model_comparison.csv', '--table2', 'reports/table2_ablation.csv', '--out', 'reports/figures'])

if __name__ == '__main__':
    main()
