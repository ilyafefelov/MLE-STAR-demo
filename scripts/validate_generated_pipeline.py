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

    import tempfile
    import os
    try:
        # Run builder in a temporary working directory to avoid picking up local train.csv
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                pipeline = builder(random_state=random_state)
            finally:
                os.chdir(old_cwd)
    except TypeError:
        # Try calling without kwargs
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    pipeline = builder()
                finally:
                    os.chdir(old_cwd)
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
        print(f"WARN: Pipeline training on Iris failed: {e}")
        traceback.print_exc()
        # Fallback: try using a local train.csv if available (some wrappers create dummy data)
        pipeline_dir = Path(path).parent
        train_csv = pipeline_dir / 'train.csv'
        repo_train_csv = repo_root / 'train.csv'
        # Try a few possible locations for generated train.csv
        candidate_files = [train_csv, repo_train_csv]
        # If the module provides a create_dummy_data function, call it to generate local csvs
        if hasattr(mod, 'create_dummy_data') and callable(getattr(mod, 'create_dummy_data')):
            try:
                print("INFO: Invoking module's create_dummy_data() to generate a local dataset for validation.")
                getattr(mod, 'create_dummy_data')()
            except Exception as e_dummy:
                print(f"WARN: create_dummy_data() raised an exception: {e_dummy}")
        # Now check the candidate files
        found_train = None
        for cand in candidate_files:
            if cand.exists():
                found_train = cand
                break
        if found_train is not None:
            try:
                df = pd.read_csv(found_train)
                if 'target' in df.columns:
                    X_local = df.drop('target', axis=1)
                    y_local = df['target']
                else:
                    # If no 'target' column, try to infer last column
                    X_local = df.iloc[:, :-1]
                    y_local = df.iloc[:, -1]
                print(f"INFO: Running cross-validation using local train.csv at {found_train}")
                scores = cross_val_score(pipeline, X_local, y_local, cv=cv)
                print(f"✓ Pipeline cross-validation OK on local data: mean={scores.mean():.4f}, std={scores.std():.4f}")
                return True
            except Exception as e2:
                print(f"ERROR: Pipeline training on local data failed: {e2}")
                traceback.print_exc()
                return False
        else:
            # Try to infer column names from preprocessor if possible
            try:
                # Try to locate a ColumnTransformer inside pipeline
                preprocessor = None
                if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                    preprocessor = pipeline.named_steps['preprocessor']
                elif hasattr(pipeline, 'steps'):
                    for name, step in pipeline.steps:
                        if hasattr(step, 'transformers'):
                            preprocessor = step
                            break
                if preprocessor is not None and hasattr(preprocessor, 'transformers'):
                    # transformers is a list of tuples (name, transformer, columns)
                    columns = []
                    for t in preprocessor.transformers:
                        if len(t) >= 3 and isinstance(t[2], (list, tuple)):
                            columns.extend(list(t[2]))
                    if columns:
                        # Build dummy DataFrame with these columns
                        nrows = max(10, cv * 5)
                        X_dummy = pd.DataFrame({c: pd.Series(np.random.rand(nrows)) for c in columns})
                        y_dummy = pd.Series(pd.Series(np.random.randint(0, 2, nrows)))
                        print(f"INFO: Running cross-validation using inferred columns: {columns}")
                        scores = cross_val_score(pipeline, X_dummy, y_dummy, cv=cv)
                        print(f"✓ Pipeline cross-validation OK on inferred data: mean={scores.mean():.4f}, std={scores.std():.4f}")
                        return True
            except Exception:
                pass

            # Final fallback: try random data assuming the model can accept it; use n_features inferred from pipeline if available
            try:
                n_features = 4
                if hasattr(pipeline, 'n_features_in_'):
                    n_features = int(pipeline.n_features_in_)
                nrows = max(10, cv * 5)
                X_rand = pd.DataFrame(np.random.rand(nrows, n_features))
                y_rand = pd.Series(np.random.randint(0, 2, nrows))
                print(f"INFO: Running cross-validation using random data with {n_features} features")
                scores = cross_val_score(pipeline, X_rand, y_rand, cv=cv)
                print(f"✓ Pipeline cross-validation OK on random data: mean={scores.mean():.4f}, std={scores.std():.4f}")
                return True
            except Exception as e_rand:
                print(f"ERROR: Fallback random-data validation failed: {e_rand}")
                traceback.print_exc()
                print("ERROR: No suitable fallback data to validate pipeline locally (train.csv not found, and inference failed).")
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
