#!/usr/bin/env python
"""Simple check: import a wrapper module and call build_full_pipeline(random_state=42).
Exits with non-zero on failures.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_path(path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception as ex:
        print(f"Error importing module {path}: {ex}")
        return None
    return module


def check_wrapper(path: Path) -> bool:
    module = load_module_from_path(path)
    if module is None:
        return False
    builder = getattr(module, 'build_full_pipeline', None)
    if builder is None:
        print(f"No build_full_pipeline found in {path}")
        return False
    try:
        p = builder(random_state=42)
    except Exception as ex:
        print(f"build_full_pipeline failed for {path}: {ex}")
        return False
    # Check that returned object is a scikit-learn like estimator (has fit) OR pipeline
    if not hasattr(p, 'fit'):
        print(f"build_full_pipeline did not return an estimator with fit for {path}; got {type(p)}")
        return False
    print(f"OK: {path} -> {type(p)}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrapper', type=str, default=None)
    parser.add_argument('--dir', type=str, default=None, help='Directory to find wrapper files')
    args = parser.parse_args()

    if args.wrapper is None and args.dir is None:
        parser.print_help()
        sys.exit(2)
    targets = []
    if args.wrapper:
        targets.append(Path(args.wrapper))
    if args.dir:
        p = Path(args.dir)
        if p.is_dir():
            targets.extend([f for f in p.iterdir() if f.suffix == '.py'])
    all_ok = True
    for t in targets:
        ok = check_wrapper(t)
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
