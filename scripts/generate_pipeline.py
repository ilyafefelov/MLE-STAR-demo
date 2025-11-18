#!/usr/bin/env python
"""
Generate a pipeline for a dataset using the MLE-STAR programmatic agent and copy the generated pipeline
into model_comparison_results as a named Python file.

Usage:
 python scripts/generate_pipeline.py --dataset iris --out model_comparison_results/gemini_live_iris.py
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

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
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to generate pipeline for (breast_cancer, wine, digits, iris)')
    parser.add_argument('--out', type=str, default=None, help='Output file path for generated pipeline')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to .env containing GEMINI_API_TOKEN or GOOGLE_API_KEY')
    parser.add_argument('--agent-script', type=str, default='scripts/run_mle_star_programmatic.py', help='Script that runs the MLE-STAR agent')
    parser.add_argument('--model', type=str, default=None, help='Root agent model to use (e.g. gemini-2.5-flash-lite)')
    args = parser.parse_args()

    env_path = Path(args.env_file)
    load_env(env_path)

    # Ensure GOOGLE_API_KEY exists (map GEMINI_API_TOKEN if present)
    if 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_TOKEN' in os.environ:
        os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']

    # Call the agent script; we expect it to generate files under machine_learning_engineering/workspace/<dataset>/
    dataset = args.dataset
    agent_script = Path(args.agent_script)
    if not agent_script.exists():
        print(f'Agent script not found: {agent_script}')
        sys.exit(1)

    print(f'Generating pipeline for dataset: {dataset}')
    import subprocess
    env = os.environ.copy()
    env['MLESTAR_TARGET_DATASET'] = dataset
    if args.model:
        env['ROOT_AGENT_MODEL'] = args.model
    # Call agent script; optionally, forward model as CLI arg when present
    agent_cmd = [sys.executable, str(agent_script)]
    if args.model:
        agent_cmd.extend(['--model', args.model])
    proc = subprocess.run(agent_cmd, env=env)
    if proc.returncode != 0:
        print('Agent script failed; check logs and API access.')
        sys.exit(proc.returncode)

    # Find generated .py files under the workspace
    workspace_dir = Path('adk-samples/python/agents/machine-learning-engineering/workspace') / dataset
    if not workspace_dir.exists():
        print(f'Workspace not found: {workspace_dir}')
        sys.exit(1)

    py_files = list(workspace_dir.glob('*.py'))
    if len(py_files) == 0:
        print(f'No Python files generated in workspace: {workspace_dir}')
        sys.exit(1)

    # Pick the largest one by size (heuristic for pipeline file)
    py_file = max(py_files, key=lambda p: p.stat().st_size)
    print(f'Found generated file: {py_file.name}')

    # Destination file
    out_path = Path(args.out) if args.out else Path('model_comparison_results') / f'gemini_live_{dataset}.py'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(py_file, out_path)
    print(f'Copied generated pipeline to {out_path}')

    # Return path
    print(out_path)

if __name__ == '__main__':
    main()
