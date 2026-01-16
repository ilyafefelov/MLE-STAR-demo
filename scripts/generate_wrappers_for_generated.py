#!/usr/bin/env python
"""
Create wrapper modules in `model_comparison_results` that expose build_full_pipeline()
for generated pipelines in `generated_pipelines`.

Usage:
  python scripts/generate_wrappers_for_generated.py --dataset iris --variant flash_lite
"""
import argparse
import importlib.util
from pathlib import Path
import re
import tempfile
import shutil

def is_importable_and_has_builder(module_path: Path):
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for candidate in ['create_model_pipeline', 'build_full_pipeline', 'create_pipeline']:
            if hasattr(mod, candidate):
                return candidate
    except Exception as e:
        print(f'Import error for {module_path}: {e}')
    return None


def try_fix_common_imports(module_path: Path) -> Path | None:
    """
    Apply simple heuristic fixes to common import mistakes in generated pipelines and write to a new fixed file.
    Returns the path to the fixed file if modifications were applied; otherwise None.
    """
    text = module_path.read_text(encoding='utf-8')
    modified = False
    # Common mistake: SimpleImputer is in sklearn.impute, not sklearn.preprocessing
    import_line_re = r"from\s+sklearn\.preprocessing\s+import\s+(.*)"
    import_match = re.search(import_line_re, text)
    if import_match:
        imported = import_match.group(1)
        # Split names and detect SimpleImputer
        names = [n.strip() for n in imported.split(',')]
        if 'SimpleImputer' in names:
            # Remove SimpleImputer from the preprocessing import and add a new import from sklearn.impute
            names.remove('SimpleImputer')
            new_preproc = ', '.join(names) if names else ''
            new_lines = []
            if new_preproc:
                new_lines.append(f'from sklearn.preprocessing import {new_preproc}')
            new_lines.append('from sklearn.impute import SimpleImputer')
            # Replace original line with new_lines
            text = re.sub(import_line_re, '\n'.join(new_lines), text)
            modified = True
    # Additional heuristics can be added here
    # Additional heuristics: ensure common imports exist for model selection and ensemble
    # If GridSearchCV or RandomizedSearchCV found in text but not imported, add import
    if 'GridSearchCV' in text and 'from sklearn.model_selection import' not in text:
        text = 'from sklearn.model_selection import GridSearchCV, RandomizedSearchCV\n' + text
        modified = True
    if 'GridSearchCV' in text and 'GridSearchCV' in text and 'from sklearn.model_selection import GridSearchCV' not in text:
        text = 'from sklearn.model_selection import GridSearchCV\n' + text
        modified = True
    if 'SVC' in text and 'from sklearn.svm import SVC' not in text:
        text = 'from sklearn.svm import SVC\n' + text
        modified = True
    if 'RandomForestClassifier' in text and 'from sklearn.ensemble import RandomForestClassifier' not in text:
        text = 'from sklearn.ensemble import RandomForestClassifier\n' + text
        modified = True
    if 'train_test_split' in text and 'from sklearn.model_selection import train_test_split' not in text:
        text = 'from sklearn.model_selection import train_test_split\n' + text
        modified = True
    if not modified:
        return None
    # Write out a temp file beside the original
    fixed_path = module_path.parent / (module_path.stem + '_fixed.py')
    fixed_path.write_text(text, encoding='utf-8')
    return fixed_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='iris', help='comma separated datasets')
    parser.add_argument('--variants', type=str, default='flash_lite,flash,pro')
    parser.add_argument('--generated-dir', type=str, default='generated_pipelines')
    parser.add_argument('--out-dir', type=str, default='model_comparison_results')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    generated_dir = Path(args.generated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        for variant in variants:
            # Prefer variant-specific file, but try cleaned and broader fallbacks too
            candidates = [
                generated_dir / f'pipeline_{ds}_{variant}.py',
                generated_dir / f'pipeline_{ds}_clean.py',
                generated_dir / f'pipeline_{ds}.py'
            ]
            gen_file = None
            for c in candidates:
                if c.exists():
                    gen_file = c
                    break
            if gen_file is None:
                print('No generated file for', ds, variant)
                continue
            if not gen_file.exists():
                print('No generated file for', ds, variant)
                continue
            wrapper_path = out_dir / f'gemini_live_{variant}_{ds}.py'
            # Attempt to import and detect builder function
            # Try to import the chosen candidate; if it fails, try other candidates
            builder = None
            for c in candidates:
                if not c.exists():
                    continue
                builder = is_importable_and_has_builder(c)
                if builder:
                    gen_file = c
                    break
                # Try small heuristics to fix common import errors
                fixed = try_fix_common_imports(c)
                if fixed:
                    builder = is_importable_and_has_builder(fixed)
                    if builder:
                        gen_file = fixed
                        break
            if not builder:
                print(f'Could not import {gen_file} or no builder found; skipping wrapper.')
                continue
            with open(wrapper_path, 'w', encoding='utf-8') as f:
                f.write(f'"""Wrapper generated by generate_wrappers_for_generated.py"""\n')
                module_name = gen_file.stem
                f.write(f'from generated_pipelines.{module_name} import {builder} as _builder\n\n')
                f.write('def build_full_pipeline(*args, **kwargs):\n')
                f.write('    """Call the original builder and ensure a single estimator is returned (not tuple)\n')
                f.write('    This handles generators that return (pipeline, metadata) or similar structures.\n')
                f.write('    """\n')
                f.write('    p = _builder(*args, **kwargs)\n')
                f.write('    # If a tuple is returned, try to find an estimator with fit()\n')
                f.write('    if isinstance(p, tuple):\n')
                f.write('        for el in p:\n')
                f.write('            if hasattr(el, "fit"):\n')
                f.write('                return el\n')
                f.write('        return p[0]\n')
                f.write('    return p\n')
            print('Created wrapper:', wrapper_path)

if __name__ == '__main__':
    main()
