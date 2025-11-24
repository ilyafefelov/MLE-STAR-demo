#!/usr/bin/env python
"""
Programmatically validate all pipeline wrappers by instantiating each ablation variant,
extracting pipeline step names, and writing a summary markdown + CSV.

Usage:
    python scripts/validate_wrappers_ablation.py --wrappers-dir model_comparison_results --out-md docs/PIPELINE_ABLATION_SUMMARY.md --out-csv docs/PIPELINE_ABLATION_SUMMARY.csv
"""
import os
import glob
import importlib.util
import sys
from pathlib import Path
import inspect
import argparse
import csv
from collections import OrderedDict

# Try to import sklearn utilities if available
try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor
    from sklearn.base import BaseEstimator
except Exception:
    Pipeline = object
    ColumnTransformer = object
    GridSearchCV = object
    VotingClassifier = object
    VotingRegressor = object
    StackingClassifier = object
    StackingRegressor = object
    BaggingClassifier = object
    BaggingRegressor = object
    BaseEstimator = object


VARIANTS = ["full", "no_scaling", "no_feature_engineering", "no_tuning", "no_ensemble", "minimal"]


def load_module_from_path(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as ex:
        # Import may execute arbitrary code; catch and keep going
        print(f"[WARN] Failed to import {path}: {ex}")
        return None
    return module


def find_builder_fn(module):
    candidates = [
        "build_full_pipeline",
        "build_pipeline",
        "build_pipeline_wrapper",
        "build",
        "build_estimator",
        "make_pipeline",
        "create_pipeline",
    ]
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    # fallback: any callable at module level that takes random_state?
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                if any(p.name in ("random_state", "seed") for p in sig.parameters.values()):
                    return attr_name, attr
            except Exception:
                continue
    return None, None


def try_build_with_variant(builder, variant):
    """Attempt to call builder with different signatures to build variant pipeline"""
    tries = []
    # typical try patterns
    tries.append({"random_state": 42, variant: True})
    tries.append({"random_state": 42, "config": {variant: True}})
    tries.append({"random_state": 42, "ablation_config": {variant: True}})
    tries.append({"random_state": 42, "ablation": {variant: True}})
    tries.append({"random_state": 42, "variant": variant})
    tries.append({"random_state": 42})
    for kw in tries:
        try:
            sig = inspect.signature(builder)
            # filter kwargs to allowed params
            filtered = {k: v for k, v in kw.items() if k in sig.parameters}
            # if config is allowed but expects dict, pass dict, else try anyway
            pipeline = builder(**filtered)
            return pipeline, filtered
        except Exception:
            continue
    # final fallback: try call with no args
    try:
        pipeline = builder()
        return pipeline, {}
    except Exception:
        raise RuntimeError("builder invocation failed for all patterns")


def extract_steps(obj):
    """Extract a readable list of steps from a Pipeline/estimator object"""
    try:
        # Pipeline
        if isinstance(obj, Pipeline):
            steps = []
            for name, step in obj.steps:
                if isinstance(step, Pipeline):
                    inner = extract_steps(step)
                    steps.append(f"{name}->({', '.join(inner)})")
                elif isinstance(step, ColumnTransformer):
                    transformers = [t[0] for t in step.transformers]
                    steps.append(f"{name}->ColumnTransformer({', '.join(transformers)})")
                elif isinstance(step, GridSearchCV):
                    est = step.estimator
                    steps.append(f"{name}->GridSearchCV({est.__class__.__name__})")
                elif isinstance(step, (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor)):
                    steps.append(f"{name}->{step.__class__.__name__}")
                elif isinstance(step, BaseEstimator):
                    steps.append(f"{name}->{step.__class__.__name__}")
                else:
                    steps.append(f"{name}->{type(step).__name__}")
            return steps
        else:
            # top-level estimator
            return [obj.__class__.__name__]
    except Exception:
        # Generic fallback: give type name
        return [type(obj).__name__]


def simulate_ablation_steps(steps, variant):
    """Given a list of step descriptions, remove or alter names according to variant heuristics"""
    keywords = {
        "no_scaling": ["scaler", "standard", "minmax", "robust", "scaling"],
        "no_feature_engineering": ["pca", "feature", "select", "poly", "transform", "fe_"],
        "no_tuning": ["gridsearch", "bayessearch", "randomsearch", "optuna", "tuner"],
        "no_ensemble": ["voting", "stacking", "bagging", "ensemble"],
        "minimal": ["pca", "feature", "select", "poly", "gridsearch", "voting", "stacking", "bagging"],
    }
    keyset = keywords.get(variant, [])
    if variant == "full":
        return steps
    filtered = []
    for s in steps:
        s_lower = s.lower()
        removed = False
        for kw in keyset:
            if kw in s_lower:
                removed = True
                break
        if not removed:
            filtered.append(s)
    # Keep at least model if everything removed
    if not filtered and steps:
        # try to keep the last element assuming it is the estimator
        filtered = [steps[-1]]
    return filtered


def main(wrapper_dir="model_comparison_results", out_md="docs/PIPELINE_ABLATION_SUMMARY.md", out_csv="docs/PIPELINE_ABLATION_SUMMARY.csv"):
    wrappers = sorted(glob.glob(os.path.join(wrapper_dir, "*_pipeline_wrapper.py")))
    if not wrappers:
        print("[ERROR] No wrappers found in", wrapper_dir)
        return
    rows = []
    # Ensure repository root is on sys.path so `src` imports work
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import ablation adapter
    try:
        from src.mle_star_ablation.mle_star_generated_pipeline import set_full_pipeline_callable, build_pipeline
        from src.mle_star_ablation.config import AblationConfig
    except Exception as e:
        print(f"[ERROR] Could not import ablation adapter: {e}")
        return

    for wrapper_path in wrappers:
        module = load_module_from_path(wrapper_path)
        if module is None:
            rows.append((wrapper_path, None, "ERROR_IMPORT", "IMPORT_FAILED", "IMPORT_FAILED"))
            continue
        bname, builder = find_builder_fn(module)
        if builder is None:
            rows.append((wrapper_path, None, "ERROR_NO_BUILDER", "NO_BUILD_FN", "NO_BUILD_FN"))
            continue
        # Register the wrapper builder in the ablation adapter so we can request accurate ablation variants
        try:
            builder_fn = getattr(module, bname)
            set_full_pipeline_callable(builder_fn)
        except Exception as e:
            print(f"[WARN] Could not register builder for {wrapper_path}: {e}")

        for variant in VARIANTS:
            try:
                # Build AblationConfig for this variant
                cfg_kwargs = {}
                if variant == 'no_scaling':
                    cfg_kwargs = {'name': variant, 'use_scaling': False}
                elif variant == 'no_feature_engineering':
                    cfg_kwargs = {'name': variant, 'use_feature_engineering': False}
                elif variant == 'no_tuning':
                    cfg_kwargs = {'name': variant, 'use_hyperparam_tuning': False}
                elif variant == 'no_ensemble':
                    cfg_kwargs = {'name': variant, 'use_ensembling': False}
                elif variant == 'minimal':
                    cfg_kwargs = {'name': variant, 'use_scaling': False, 'use_feature_engineering': False, 'use_hyperparam_tuning': False, 'use_ensembling': False}
                else:
                    cfg_kwargs = {'name': variant}
                config = AblationConfig(**cfg_kwargs)
                pipeline = build_pipeline(config, deterministic=True, random_state=42)
                steps = extract_steps(pipeline)
                steps_sim = steps
                note = 'ADAPTER'
            except Exception as ex:
                # Fallback to prior approach
                try:
                    pipeline, used_kwargs = try_build_with_variant(builder, variant)
                    steps = extract_steps(pipeline)
                    steps_sim = simulate_ablation_steps(steps, variant)
                    note = 'SIMULATED_FALLBACK'
                except Exception as ex2:
                    steps_sim = [f"BUILD_ERROR: {ex2}"]
                    note = 'ERROR_BUILD'
            rows.append((os.path.basename(wrapper_path), bname, variant, "; ".join(steps_sim), note))
    # Write CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wrapper_path", "builder_name", "variant", "steps", "note"])
        for r in rows:
            writer.writerow(r)
    # Write Markdown
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# Pipeline Ablation Summary\n\n")
        fh.write("| wrapper_path | builder | variant | steps | note |\n")
        fh.write("|---|---|---|---|---|\n")
        for wrapper_path, bname, variant, steps, note in rows:
            safe_steps = steps.replace("|", "\\|") if isinstance(steps, str) else steps
            fh.write(f"| `{wrapper_path}` | `{bname}` | `{variant}` | `{safe_steps}` | `{note}` |\n")
    print("Wrote CSV:", out_csv)
    print("Wrote MD:", out_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrappers-dir", default="model_comparison_results")
    parser.add_argument("--out-md", default="docs/PIPELINE_ABLATION_SUMMARY.md")
    parser.add_argument("--out-csv", default="docs/PIPELINE_ABLATION_SUMMARY.csv")
    args = parser.parse_args()
    main(args.wrappers_dir, args.out_md, args.out_csv)
