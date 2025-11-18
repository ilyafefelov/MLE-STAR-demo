#!/usr/bin/env python
"""
Check whether a live pipeline generation created an expected pipeline file and workspace contents.

Usage:
  python scripts/check_generation_success.py --dataset iris --variant flash_lite --pipeline model_comparison_results/gemini_live_flash_lite_iris.py

This script checks:
 - if the ADK dependency 'google.adk' is importable
 - if the expected workspace for the dataset exists and contains generated Python files
 - if the target pipeline file exists and has at least some content
"""
import argparse
import sys
from pathlib import Path


def check_adk_importable():
    try:
        import google.adk  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--variant', default=None, help='Optional variant used for naming of pipeline file')
    parser.add_argument('--pipeline', default=None, help='Path to expected generated pipeline file (optional)')
    args = parser.parse_args()

    ok, err = check_adk_importable()
    if not ok:
        print(f"ERROR: ADK not importable: {err}")
        print("Install ADK Python library per adk-samples instructions (https://google.github.io/adk-docs/get-started/installation/#python)")
    else:
        print("OK: ADK 'google.adk' importable")

    workspace_dir = Path('adk-samples/python/agents/machine-learning-engineering/machine_learning_engineering/workspace') / args.dataset
    if workspace_dir.exists():
        py_files = list(workspace_dir.glob('*.py'))
        print(f"Workspace dir found: {workspace_dir} with {len(py_files)} python files")
        for p in py_files:
            size = p.stat().st_size
            print(f" - {p.name} size={size}")
    else:
        print(f"Workspace dir not found: {workspace_dir}")

    if args.pipeline:
        pipeline_path = Path(args.pipeline)
        if pipeline_path.exists():
            print(f"Pipeline file exists: {pipeline_path} size={pipeline_path.stat().st_size}")
            with pipeline_path.open('r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(1024)
                print('--- pipeline sample ---')
                print(sample.strip())
        else:
            print(f"Pipeline file not found: {pipeline_path}")

    # return nonzero if pipeline not found or adk not present
    if (not ok) or (not workspace_dir.exists()) or (args.pipeline and not Path(args.pipeline).exists()):
        sys.exit(1)
    print('Checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python
"""
Check that generated pipeline files exist and are importable, and they expose a build_full_pipeline or build_pipeline function.

Usage:
  python scripts/check_generation_success.py --file model_comparison_results/gemini_live_iris.py
  python scripts/check_generation_success.py --dir model_comparison_results/
"""

import argparse
from pathlib import Path
import importlib.util
import sys


def check_file(file_path: Path):
    if not file_path.exists():
        return False, f'File not found: {file_path}'
    try:
        spec = importlib.util.spec_from_file_location('generated_pipeline', str(file_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return False, f'Could not import {file_path}: {e}'

    # Check for functions commonly used by our codebase
    if hasattr(mod, 'build_full_pipeline') or hasattr(mod, 'build_pipeline') or hasattr(mod, 'build'):
        return True, 'OK'
    else:
        return False, 'No build_full_pipeline/build_pipeline/build function found in module'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    if not args.file and not args.dir:
        print('Provide --file or --dir to check')
        sys.exit(2)

    if args.file:
        ok, msg = check_file(Path(args.file))
        print(f'{args.file}: {ok} - {msg}')
        if not ok:
            sys.exit(1)
        return

    # directory check
    p = Path(args.dir)
    if not p.exists():
        print(f'Directory does not exist: {p}')
        sys.exit(2)

    failures = []
    for f in p.glob('*.py'):
        ok, msg = check_file(f)
        print(f'{f.name}: {ok} - {msg}')
        if not ok:
            failures.append((f.name, msg))

    if failures:
        print('\nSome pipeline files are not valid:')
        for n, m in failures:
            print(f'  - {n}: {m}')
        sys.exit(1)
    else:
        print('\nAll pipeline files in directory appear valid.')

if __name__ == '__main__':
    main()
