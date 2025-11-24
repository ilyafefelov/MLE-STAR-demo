#!/usr/bin/env python
"""Simple tests for auto_make_pipeline_wrapper AST-based estimator detection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from auto_make_pipeline_wrapper import detect_estimator_class


def test_cases():
    cases = [
        ("model = lgb.LGBMRegressor(n_estimators=100)", ('LGBMRegressor', 'regressor')),
        ("clf = RandomForestClassifier(n_estimators=50)", ('RandomForestClassifier', 'classifier')),
        ("model = XGBRegressor(objective='reg:squarederror')", ('XGBRegressor', 'regressor')),
        ("model = CatBoostRegressor(iterations=10)", ('CatBoostRegressor', 'regressor')),
        ("model = CatBoostClassifier(iterations=10)", ('CatBoostClassifier', 'classifier')),
        ("clf = LogisticRegression()", ('LogisticRegression', 'classifier')),
        ("# Nothing here", (None, None)),
        ("some = MyCustomEstimator()", (None, None)),
    ]

    passed = 0
    for text, expected in cases:
        got = detect_estimator_class(text)
        ok = got == expected
        print(f"Input: {text!r} -> expected {expected}, got {got} -> {'OK' if ok else 'FAIL'}")
        if ok:
            passed += 1

    print(f"\nPassed {passed}/{len(cases)} cases")


if __name__ == '__main__':
    test_cases()
    # Now test extracted function wrapper
    import subprocess
    import os
    out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_comparison_results', 'tmp_with_build_wrapper.py')
    # Remove if exists
    try:
        os.remove(out_file)
    except Exception:
        pass
    ret = subprocess.run([sys.executable, 'auto_make_pipeline_wrapper.py', '--src', 'tmp_with_build.py', '--out', out_file], cwd=os.path.dirname(__file__))
    if ret.returncode == 0:
        print('\nWrapper generated:', out_file)
        with open(out_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print('\nWrapper content excerpt:\n')
            print('\n'.join(content.splitlines()[:40]))
    else:
        print('\nWrapper generation failed (exit code', ret.returncode, ')')
