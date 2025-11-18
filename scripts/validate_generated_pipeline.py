#!/usr/bin/env python
"""
Validate a generated pipeline file or wrapper by importing it and running a minimal test.

This script is intended to be used as a preflight check for generated pipelines.
It verifies that `build_full_pipeline(random_state=42)` exists, returns a scikit-learn
estimator/Pipeline (i.e., has `.fit()`), and runs a very small cross-validation to ensure
the pipeline trains without errors.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import traceback

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import pandas as pd


def validate_pipeline_file(path: Path, cv=2, random_state=42) -> bool:
    print(f"Validating pipeline: {path}")
    # Ensure repository root is in sys.path so modules like generated_pipelines import correctly
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR: Could not import module {path}: {e}")
        traceback.print_exc()
        return False

    # Find possible builder functions
    candidates = ['build_full_pipeline', 'create_model_pipeline', 'create_pipeline']
    builder = None
    for cand in candidates:
        if hasattr(mod, cand):
            builder = getattr(mod, cand)
            break

    if builder is None:
        print("ERROR: No builder function found (tried names: build_full_pipeline, create_model_pipeline, create_pipeline)")
        return False

    try:
        pipeline = builder(random_state=random_state)
    except TypeError:
        # Try calling without kwargs
        try:
            pipeline = builder()
        except Exception as e:
            print(f"ERROR: builder invocation failed: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"ERROR: builder invocation raised: {e}")
        traceback.print_exc()
        return False

    # Ensure pipeline-like
    if not hasattr(pipeline, 'fit'):
        print("ERROR: builder did not return an estimator with fit() method")
        return False

    # Quick cross_val_score on iris to ensure pipeline trains
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    try:
        scores = cross_val_score(pipeline, X[:120], y[:120], cv=cv)
        print(f"✓ Pipeline cross-validation OK: mean={scores.mean():.4f}, std={scores.std():.4f}")
        return True
    except Exception as e:
        print(f"ERROR: Pipeline training triggered error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline-file', type=str, required=True, help='Path to generated pipeline or wrapper file')
    parser.add_argument('--cv', type=int, default=2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    path = Path(args.pipeline_file)
    if not path.exists():
        print(f"Pipeline file not found: {path}")
        sys.exit(2)

    ok = validate_pipeline_file(path, cv=args.cv, random_state=args.random_state)
    if not ok:
        print("Validation FAILED")
        sys.exit(1)
    else:
        print("Validation SUCCESS")
        sys.exit(0)


if __name__ == '__main__':
    main()
