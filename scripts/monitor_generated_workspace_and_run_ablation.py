#!/usr/bin/env python
"""
Monitor MLE-STAR ADK workspace for a generated pipeline and automatically
copy to `model_comparison_results`, validate, and run ablation once ready.

Usage:
  python scripts/monitor_generated_workspace_and_run_ablation.py --dataset iris --timeout 3600 --n-runs 3 --deterministic
"""
import argparse
import time
from pathlib import Path
import shutil
import subprocess
import os
import sys
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=3600, help='Max seconds to wait')
    parser.add_argument('--interval', type=int, default=30, help='Poll interval seconds')
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--env-file', type=str, default='.env')
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    repo_root = Path(__file__).parent.parent
    workspace_base = repo_root / 'adk-samples' / 'python' / 'agents' / 'machine-learning-engineering' / 'machine_learning_engineering' / 'workspace'
    workspace_dir = workspace_base / ds
    timeout_at = time.time() + args.timeout

    print(f'Waiting for ADK workspace pipeline in: {workspace_dir}')
    while time.time() < timeout_at:
        # Sometimes agent names the folder with task variations; look for directories that contain dataset as prefix
        candidate_dirs = [p for p in workspace_base.iterdir() if p.is_dir() and p.name.startswith(ds)]
        py_files = []
        for candidate in candidate_dirs:
            # Search recursively for pipeline files, agent may write them into nested subfolders
            py_files.extend(list(candidate.rglob('*.py')))
            if py_files:
                # use largest file heuristic
                best = max(py_files, key=lambda p: p.stat().st_size)
                print(f'Found candidate generated file: {best}')
                out_file = repo_root / 'model_comparison_results' / f'gemini_live_{ds}.py'
                out_file.parent.mkdir(parents=True, exist_ok=True)
                # Only copy if out_file doesn't exist or source is newer
                if out_file.exists():
                    if out_file.stat().st_mtime >= best.stat().st_mtime:
                        print(f'Existing destination {out_file} is up to date (skipping)')
                        continue
                    else:
                        print(f'Destination {out_file} is older than {best}, overwriting')
                shutil.copy(best, out_file)
                print(f'Copied {best.name} to {out_file}')
                # Create an in-repo wrapper that safely imports the generated module and exposes build_full_pipeline
                wrapper_path = out_file.parent / f'gemini_live_{ds}_wrapped.py'
                wrapper_text = f"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('gemini_live_{ds}', r'{out_file}')
mod = importlib.util.module_from_spec(spec)
sys.modules['gemini_live_{ds}'] = mod
spec.loader.exec_module(mod)

builder = getattr(mod, 'build_full_pipeline', None) or getattr(mod, 'create_model_pipeline', None) or getattr(mod, 'create_pipeline', None)

def build_full_pipeline(*args, **kwargs):
    if builder is None:
        raise RuntimeError('Could not find builder function in generated module')
    p = builder(*args, **kwargs)
    if isinstance(p, tuple):
        for el in p:
            if hasattr(el, 'fit'):
                return el
        return p[0]
    return p
"""
                try:
                    wrapper_path.write_text(wrapper_text, encoding='utf-8')
                    print(f'Created safe wrapper at {wrapper_path}')
                except Exception as e:
                    print(f'Could not write wrapper: {e}')
                # Inspect
                subprocess.run([sys.executable, str(repo_root / 'scripts' / 'inspect_generated_pipelines.py'), '--dir', str(candidate)])
                # Validate
                # Validate the copy or the wrapper if created
                pipeline_to_validate = str(wrapper_path) if 'wrapper_path' in locals() and wrapper_path.exists() else str(out_file)
                subprocess.run([sys.executable, str(repo_root / 'scripts' / 'validate_generated_pipeline.py'), '--pipeline-file', pipeline_to_validate, '--cv', '2', '--random-state', '42'])
                # Run ablation
                cmd = [sys.executable, str(repo_root / 'scripts' / 'run_ablation.py'), '--dataset', ds, '--n-runs', str(args.n_runs), '--pipeline-file', pipeline_to_validate]
                if args.deterministic:
                    cmd.extend(['--deterministic', '--seed', '42'])
                cmd.append('--no-plots')
                cmd.append('--verbose')
                print('Running ablation:', ' '.join(cmd))
                subprocess.run(cmd)
                print('Ablation flow completed for dataset:', ds)
                # Write a monitor state file so other processes/CI/agents can pick up status
                monitor_dir = repo_root / 'monitor_logs'
                monitor_dir.mkdir(parents=True, exist_ok=True)
                result_json_path = monitor_dir / f'{ds}_last_ablation.json'
                payload = {
                    'dataset': ds,
                    'pipeline_file': pipeline_to_validate,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                }
                try:
                    result_json_path.write_text(json.dumps(payload), encoding='utf-8')
                    print(f'Wrote monitor completion state to {result_json_path}')
                except Exception as e:
                    print('Could not write monitor state file:', e)
                return
        time.sleep(args.interval)
    print('Timeout reached, no generated pipeline found in workspace')
    sys.exit(2)


if __name__ == '__main__':
    main()
