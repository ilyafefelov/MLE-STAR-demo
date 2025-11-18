#!/usr/bin/env python
"""
Inspect generated pipelines (both ADK and direct Gemini) to ensure they include
`preprocessor`, `feature_engineering`, `model` and optional components needed
for ablation experiments (tuning/ensembling).

Usage:
  python scripts/inspect_generated_pipelines.py --dir generated_pipelines
"""
import argparse
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='generated_pipelines', help='Directory with generated pipeline .py files')
    return parser.parse_args()


def inspect_files(dir_path: Path):
    files = list(dir_path.glob('*.py'))
    if not files:
        print(f'No pipeline files found in {dir_path}')
        return
    for f in files:
        try:
            content = f.read_text(encoding='utf-8')
        except Exception:
            content = ''
        required = [
            ('preprocessor', 'preprocessor'),
            ('feature_engineering', 'feature_engineering'),
            ('model', 'model')
        ]
        optional = [
            ('tuning', 'GridSearchCV'),
            ('tuning2', 'RandomizedSearchCV'),
            ('ensemble_voting', 'VotingClassifier'),
            ('ensemble_stacking', 'StackingClassifier'),
            ('ensemble_bagging', 'BaggingClassifier')
        ]
        missing_required = [name for name, token in required if token not in content]
        found_optional = [name for name, token in optional if token in content]
        print(f'=== {f.name} ===')
        if missing_required:
            print(f'  ⚠️ Missing required steps: {missing_required}')
        else:
            print('  ✅ Required steps present')
        if found_optional:
            print(f'  ✅ Optional: {found_optional}')
        else:
            print('  ℹ️ Optional components (tuning/ensemble) not detected; ablation configs might be no-ops')


def main():
    args = parse_args()
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f'Directory not found: {dir_path}')
        return
    inspect_files(dir_path)


if __name__ == '__main__':
    main()
