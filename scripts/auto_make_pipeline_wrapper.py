#!/usr/bin/env python
"""
Heuristic wrapper creator: for a training-style generated script that trains a model (no pipeline),
this script creates a safe wrapper that returns a scikit-learn Pipeline with a minimal preprocessor
(SimpleImputer + StandardScaler) and the detected estimator class (unfitted) using `random_state`.

Usage:
  python scripts/auto_make_pipeline_wrapper.py --src model_comparison_results/gemini_live_iris.py --out model_comparison_results/gemini_live_iris_wrapper_auto.py
"""
import argparse
import ast
from pathlib import Path

ESTIMATOR_CLASSES = [
    'HistGradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'SVC', 'XGBClassifier'
]

def detect_estimator_class(text: str):
    for cls in ESTIMATOR_CLASSES:
        if cls in text:
            return cls
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('Source not found:', src)
        return 1

    text = src.read_text(encoding='utf-8')
    cls = detect_estimator_class(text)
    if cls is None:
        print('Could not detect known estimator class in source; defaulting to RandomForestClassifier')
        cls = 'RandomForestClassifier'

    # Build wrapper content
    imports = [
        'from sklearn.pipeline import Pipeline',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.impute import SimpleImputer',
        f'from sklearn.ensemble import {cls}' if cls in ('RandomForestClassifier', 'HistGradientBoostingClassifier') else f'from sklearn.linear_model import {cls}' if cls == 'LogisticRegression' else f'from sklearn.svm import {cls}',
        '\n'
    ]
    content = []
    content.extend(imports)
    content.append('\ndef build_full_pipeline(random_state=42):')
    content.append('    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])')
    if cls == 'HistGradientBoostingClassifier':
        content.append(f'    model = {cls}(random_state=random_state)')
    else:
        content.append(f'    model = {cls}()')
    content.append('    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])')
    content.append('    return pipeline')

    out.write_text('\n'.join(content), encoding='utf-8')
    print('Wrote auto pipeline wrapper with estimator:', cls, '->', out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
