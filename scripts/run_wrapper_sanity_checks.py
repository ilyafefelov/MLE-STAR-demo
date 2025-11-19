#!/usr/bin/env python
"""
Run runtime sanity checks for each pipeline wrapper and ablation variant.

For each wrapper:
- Register its build_full_pipeline in the adapter
- For each ablation config (full/no_scaling/no_feature_engineering/no_tuning/no_ensemble/minimal):
  - Build an adapted pipeline
  - Fit on a small subset of the dataset (use DatasetLoader to get train/test), limiting rows to avoid long runs
  - Call predict on a small test subset
  - Record result (success/failure, exception)

This script writes results to `docs/WRAPPER_SANITY_RESULTS.csv` and `docs/WRAPPER_SANITY_RESULTS.md`.

Usage:
    python scripts/run_wrapper_sanity_checks.py --config configs/experiment_suite.yaml --wrappers model_comparison_results --out docs/WRAPPER_SANITY_RESULTS
"""
import argparse
import glob
import importlib.util
import sys
import os
import traceback
import time
import csv
from pathlib import Path
import yaml

# Ensure repository root on sys.path so src imports work
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.mle_star_ablation.config import get_standard_configs
from src.mle_star_ablation.datasets import load_dataset, list_available_datasets, DatasetLoader
from src.mle_star_ablation.mle_star_generated_pipeline import set_full_pipeline_callable, build_pipeline

DEFAULT_MAX_TRAIN = 200
DEFAULT_MAX_TEST = 50


def load_module_from_path(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as ex:
        print(f"[WARN] Failed to import {path}: {ex}")
        return None
    return module


def find_builder_fn(module):
    # Popular names for builder
    for name in ('build_full_pipeline', 'build_pipeline', 'build'):
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    # Fallback: first callable
    for name in dir(module):
        if name.startswith('_'):
            continue
        attr = getattr(module, name)
        if callable(attr):
            return name, attr
    return None, None


def get_dataset_map_from_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {}
    data = yaml.safe_load(Path(manifest_path).read_text(encoding='utf-8')) or {}
    exp_map = {}
    for exp in data.get('experiments', []):
        pipeline_file = exp.get('pipeline_file')
        if not pipeline_file:
            continue
        exp_map[Path(pipeline_file).name] = {
            'dataset': exp.get('dataset'),
            'task_type': exp.get('task_type')
        }
    return exp_map


def try_fit_predict(pipeline, X_train, y_train, X_test, y_test, max_train=DEFAULT_MAX_TRAIN, max_test=DEFAULT_MAX_TEST):
    # Limit sizes
    import numpy as np
    n_train = min(max_train, len(X_train))
    n_test = min(max_test, len(X_test))
    if n_train <= 0 or n_test <= 0:
        return False, 'Empty dataset', None
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    X_te = X_test[:n_test]
    y_te = y_test[:n_test]
    # Fit/predict
    start = time.time()
    try:
        pipeline.fit(X_tr, y_tr)
    except Exception as e:
        return False, f'FitError: {e}', traceback.format_exc()
    t_fit = time.time() - start
    try:
        y_pred = pipeline.predict(X_te)
    except Exception as e:
        return False, f'PredictError: {e}', traceback.format_exc()
    t_pred = time.time() - start - t_fit
    return True, f'OK (fit {t_fit:.2f}s, pred {t_pred:.2f}s)', None


def main(manifest='configs/experiment_suite.yaml', wrappers_dir='model_comparison_results', out_base='docs/WRAPPER_SANITY_RESULTS'):
    wrapper_paths = sorted(glob.glob(os.path.join(wrappers_dir, '*_pipeline_wrapper.py')))

    dataset_map = get_dataset_map_from_manifest(manifest)

    results = []
    configs = get_standard_configs()

    for wrapper_path in wrapper_paths:
        wrapper_name = os.path.basename(wrapper_path)
        module = load_module_from_path(wrapper_path)
        if module is None:
            results.append([wrapper_name, '', 'IMPORT_ERROR', '', ''])
            continue
        bname, builder = find_builder_fn(module)
        if builder is None:
            results.append([wrapper_name, '', 'NO_BUILDER', '', ''])
            continue
        # Register in adapter
        try:
            set_full_pipeline_callable(builder)
        except Exception as e:
            results.append([wrapper_name, bname or '', 'REG_FAIL', '', str(e)])
            continue

        # Determine dataset & type from manifest map if present
        if wrapper_name in dataset_map:
            dataset_name = dataset_map[wrapper_name]['dataset']
            task_type = dataset_map[wrapper_name]['task_type']
        else:
            # heuristics
            if 'california' in wrapper_name:
                dataset_name = 'california-housing-prices'
                task_type = 'regression'
            elif 'diabetes' in wrapper_name:
                dataset_name = 'diabetes'
                task_type = 'regression'
            elif 'digits' in wrapper_name:
                dataset_name = 'digits'
                task_type = 'classification'
            elif 'iris' in wrapper_name:
                dataset_name = 'iris'
                task_type = 'classification'
            elif 'wine' in wrapper_name:
                dataset_name = 'wine'
                task_type = 'classification'
            elif 'breast' in wrapper_name or 'breast_cancer' in wrapper_name:
                dataset_name = 'breast_cancer'
                task_type = 'classification'
            else:
                # default to iris
                dataset_name = 'iris'
                task_type = 'classification'

        # load dataset once per wrapper
        try:
            X_train, X_test, y_train, y_test = load_dataset(dataset_name)
        except Exception as e:
            results.append([wrapper_name, bname, 'DATASET_FAIL', dataset_name, str(e)])
            continue

        for config in configs:
            cfg_name = config.get_name()
            # Call adapter build_pipeline to obtain ablated pipeline
            try:
                pipe = build_pipeline(config, deterministic=True, random_state=42)
            except Exception as e:
                results.append([wrapper_name, bname, cfg_name, 'BUILD_FAIL', str(e)])
                continue

            # Try fit/predict with limited set
            try:
                success, msg, tb = try_fit_predict(pipe, X_train, y_train, X_test, y_test, DEFAULT_MAX_TRAIN, DEFAULT_MAX_TEST)
                results.append([wrapper_name, bname, cfg_name, 'OK' if success else 'FAIL', msg if tb is None else tb.splitlines()[-1][:200]])
            except Exception as e:
                results.append([wrapper_name, bname, cfg_name, 'EXC', str(e)])

    # Write CSV
    out_csv = f'{out_base}.csv'
    out_md = f'{out_base}.md'
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['wrapper', 'builder', 'config', 'status', 'message'])
        for r in results:
            writer.writerow(r)
    # Write MD (table)
    with open(out_md, 'w', encoding='utf-8') as fh:
        fh.write('# Wrapper Sanity Check Results\n\n')
        fh.write('| wrapper | builder | config | status | message |\n')
        fh.write('|---|---|---|---|---|\n')
        for r in results:
            wrapper, builder, cfg, status, msg = r
            safe_msg = (msg or '').replace('|', '\\|')
            fh.write(f'| `{wrapper}` | `{builder}` | `{cfg}` | `{status}` | `{safe_msg}` |\n')
    print('Wrote results:', out_csv, out_md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_suite.yaml')
    parser.add_argument('--wrappers', default='model_comparison_results')
    parser.add_argument('--out', default='docs/WRAPPER_SANITY_RESULTS')
    parser.add_argument('--max-train', default=DEFAULT_MAX_TRAIN, type=int)
    parser.add_argument('--max-test', default=DEFAULT_MAX_TEST, type=int)
    args = parser.parse_args()
    DEFAULT_MAX_TRAIN = args.max_train
    DEFAULT_MAX_TEST = args.max_test
    main(args.config, args.wrappers, args.out)
