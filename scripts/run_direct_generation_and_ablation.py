#!/usr/bin/env python
"""
Orchestrate direct pipeline generation (simple direct SDK) and run ablation experiments for each generated pipeline.

Workflow:
 - Generate direct pipelines via `generate_direct_pipelines.py` (calls `simple_mle_star.py`)
 - Generate wrappers in `model_comparison_results/`
 - Run ablation for each dataset/variant using wrapper as pipeline-file

Usage:
  python scripts/run_direct_generation_and_ablation.py --datasets iris,wine --variants flash_lite,flash,pro --n-runs 1
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='iris,breast_cancer,wine,digits')
    parser.add_argument('--variants', type=str, default='flash_lite,flash,pro')
    parser.add_argument('--n-runs', type=int, default=20)
    parser.add_argument('--env-file', type=str, default='.env')
    parser.add_argument('--out-dir', type=str, default='generated_pipelines')
    parser.add_argument('--pipeline-out', type=str, default='model_comparison_results')
    parser.add_argument('--python', type=str, default=sys.executable)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]

    # 1) generate pipelines
    gen_cmd = [args.python, 'scripts/generate_direct_pipelines.py', '--datasets', ','.join(datasets), '--variants', ','.join(variants), '--out-dir', args.out_dir, '--env-file', args.env_file]
    print('Generate pipelines command:', ' '.join(gen_cmd))
    if subprocess.run(gen_cmd).returncode != 0:
        print('Pipeline generation step failed; aborting')
        sys.exit(1)

    # 2) create wrappers
    wrap_cmd = [args.python, 'scripts/generate_wrappers_for_generated.py', '--datasets', ','.join(datasets), '--variants', ','.join(variants), '--generated-dir', args.out_dir, '--out-dir', args.pipeline_out]
    print('Create wrappers command:', ' '.join(wrap_cmd))
    if subprocess.run(wrap_cmd).returncode != 0:
        print('Wrapper generation failed; aborting')
        sys.exit(1)

    # 3) run ablation per dataset-variant
    for ds in datasets:
        for variant in variants:
            pipeline_file = Path(args.pipeline_out) / f'gemini_live_{variant}_{ds}.py'
            if not pipeline_file.exists():
                print('Pipeline file not found, skipping:', pipeline_file)
                continue
            outdir = Path('results') / f'live_{ds}_{variant}_n{args.n_runs}'
            outdir.mkdir(parents=True, exist_ok=True)
            cmd = [args.python, 'scripts/run_ablation.py', '--dataset', ds, '--n-runs', str(args.n_runs), '--pipeline-file', str(pipeline_file), '--output-dir', str(outdir), '--verbose']
            print('Running ablation:', ' '.join(cmd))
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print('Ablation failed for', ds, variant)

if __name__ == '__main__':
    main()
