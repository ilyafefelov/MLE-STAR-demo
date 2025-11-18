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

CLASSIFIERS = [
    'HistGradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'SVC', 'XGBClassifier', 'CatBoostClassifier', 'GradientBoostingClassifier'
]
REGRESSORS = [
    'LGBMRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor', 'CatBoostRegressor'
]

FALLBACK_PARAM_GRIDS = {
    'RandomForestClassifier': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'RandomForestRegressor': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'HistGradientBoostingClassifier': {'max_depth': [3, None], 'learning_rate': [0.05, 0.1]},
    'HistGradientBoostingRegressor': {'max_depth': [3, None], 'learning_rate': [0.05, 0.1]},
    'GradientBoostingClassifier': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
    'GradientBoostingRegressor': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
    'LogisticRegression': {'C': [0.1, 1.0], 'solver': ['lbfgs']},
    'SVC': {'C': [0.5, 1.0], 'gamma': ['scale', 'auto']},
    'XGBClassifier': {'n_estimators': [200, 400], 'max_depth': [4, 6]},
    'XGBRegressor': {'n_estimators': [200, 400], 'learning_rate': [0.05, 0.1]},
    'CatBoostClassifier': {'depth': [4, 6], 'learning_rate': [0.03, 0.1]},
    'CatBoostRegressor': {'depth': [4, 6], 'learning_rate': [0.03, 0.1]},
    'LGBMRegressor': {'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1]},
}

import ast


class EstimatorVisitor(ast.NodeVisitor):
    """AST visitor to detect estimator class names used in the source file.
    It collects class names (string) from Call nodes where the callee is a Name or Attribute.
    """
    def __init__(self):
        super().__init__()
        self.names = set()

    def visit_Call(self, node: ast.Call):
        # function call like RandomForestClassifier(...)
        func = node.func
        if isinstance(func, ast.Name):
            self.names.add(func.id)
        elif isinstance(func, ast.Attribute):
            # attribute call like lgb.LGBMRegressor(...)
            self.names.add(func.attr)
        # Continue traversal
        self.generic_visit(node)


def detect_estimator_class(text: str):
    """Parse text using AST and detect estimator classes present as call targets.
    Returns (class_name, 'regressor' or 'classifier') or (None, None) if not found.
    """
    try:
        tree = ast.parse(text)
    except Exception:
        # Fallback to substring search if AST parsing fails
        for cls in REGRESSORS:
            if cls in text:
                return cls, 'regressor'
        for cls in CLASSIFIERS:
            if cls in text:
                return cls, 'classifier'
        return None, None

    visitor = EstimatorVisitor()
    visitor.visit(tree)
    names = visitor.names

    # Check for regressors first
    for cls in REGRESSORS:
        if cls in names:
            return cls, 'regressor'
    for cls in CLASSIFIERS:
        if cls in names:
            return cls, 'classifier'
    # Fallback: try substring presence for additional robustness
    for cls in REGRESSORS:
        if cls in text:
            return cls, 'regressor'
    for cls in CLASSIFIERS:
        if cls in text:
            return cls, 'classifier'
    return None, None


def extract_estimator_call_and_kwargs(text: str):
    """Extract first estimator Call node and reconstruct class name + kwargs string.
    Returns (class_name, kwarg_str) or (None, None).
    """
    try:
        tree = ast.parse(text)
    except Exception:
        return None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # get function name
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            if name in REGRESSORS or name in CLASSIFIERS:
                # build kwargs string
                kw_strings = []
                for kw in node.keywords:
                    try:
                        val_src = ast.unparse(kw.value)
                    except Exception:
                        val_src = None
                    if val_src is None:
                        continue
                    if kw.arg == 'random_state':
                        # prefer to harness wrapper's random_state param
                        continue
                    kw_strings.append(f"{kw.arg}={val_src}")
                kw_str = ', '.join(kw_strings)
                return name, kw_str
    return None, None


def extract_pipeline_steps(text: str):
    """Extract steps from a top-level Pipeline([...]) call if present.
    Returns list of steps: [{'name': 'preprocessor', 'estimator': 'Pipeline', 'estimator_name': None, 'kwargs_str': None, 'nested_steps': [...]}, {...}]
    """
    try:
        tree = ast.parse(text)
    except Exception:
        return None
    steps_found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name == 'Pipeline':
                # try to extract the first arg, a list of steps
                if node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.List):
                        step_list = []
                        for elt in arg0.elts:
                            if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                                # first elt is name (Constant/Str)
                                s_name_node = elt.elts[0]
                                try:
                                    if isinstance(s_name_node, ast.Constant):
                                        sname = s_name_node.value
                                    elif isinstance(s_name_node, ast.Str):
                                        sname = s_name_node.s
                                    else:
                                        continue
                                except Exception:
                                    continue
                                # second elt is a Call to estimator or a Name
                                est_node = elt.elts[1]
                                est_name = None
                                kw_str = None
                                nested_steps = None
                                if isinstance(est_node, ast.Call):
                                    f = est_node.func
                                    if isinstance(f, ast.Name):
                                        est_name = f.id
                                    elif isinstance(f, ast.Attribute):
                                        est_name = f.attr
                                    # collect kwargs
                                    kw_pairs = []
                                    for kw in est_node.keywords:
                                        try:
                                            val_src = ast.unparse(kw.value)
                                        except Exception:
                                            val_src = None
                                        if val_src is not None:
                                            if kw.arg == 'random_state':
                                                continue
                                            kw_pairs.append(f"{kw.arg}={val_src}")
                                    kw_str = ', '.join(kw_pairs)
                                elif isinstance(est_node, ast.Name):
                                    est_name = est_node.id
                                elif isinstance(est_node, ast.Call) and isinstance(est_node.func, ast.Name) and est_node.func.id == 'Pipeline':
                                    # nested pipeline; handle below
                                    nested_steps = []
                                # detect nested pipeline assignment structural cases, not fully exhaustive
                                step_list.append({'name': sname, 'estimator': est_node, 'estimator_name': est_name, 'kwargs_str': kw_str, 'nested_steps': nested_steps})
                        if step_list:
                            steps_found = step_list
                            break
    return steps_found


def extract_build_fn_and_imports(text: str):
    """Extract build_full_pipeline function source and top-level imports from text.
    Returns tuple (function_source_str or None, list_of_import_lines, remaining_text_lines)
    """
    try:
        tree = ast.parse(text)
    except Exception:
        return None, [], text.splitlines(), {}

    import_lines = []
    func_node = None
    # Collect import statements and locate build_full_pipeline
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Extract lines
            try:
                start = node.lineno - 1
                end = node.end_lineno
                import_lines.extend(text.splitlines()[start:end])
            except Exception:
                # Fallback: build a simple import string
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        import_lines.append(f"from {node.module} import {alias.name}")
                else:
                    for alias in node.names:
                        import_lines.append(f"import {alias.name}")
        elif isinstance(node, ast.FunctionDef) and node.name == 'build_full_pipeline':
            func_node = node

    if func_node is None:
        return None, import_lines, text.splitlines(), {}

    # Extract function source lines
    lines = text.splitlines()
    start = func_node.lineno - 1
    end = func_node.end_lineno
    fn_lines = lines[start:end]

    # Adjust function signature to ensure random_state param present
    import re
    header = fn_lines[0]
    if 'random_state' not in header:
        # If the function has no parameters: def build_full_pipeline(): -> def build_full_pipeline(random_state=42):
        m = re.match(r"(\s*def\s+build_full_pipeline\s*\()([^)]*)(\)\s*:.*)", header)
        if m:
            before, params, after = m.groups()
            params = params.strip()
            if params == '':
                new_header = f"{before}random_state: int = 42{after}"
            else:
                new_header = f"{before}random_state: int = 42, {params}{after}"
            fn_lines[0] = new_header

    # Also collect FunctionDef nodes for potential dependencies
    # Build a mapping of top-level function names to their source in the file
    func_defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            func_defs[node.name] = '\n'.join(lines[start:end])

    return '\n'.join(fn_lines), import_lines, lines, func_defs


def get_fallback_param_grid(cls_name: str | None, cls_type: str | None):
    """Return a minimal safe param grid tailored to the detected estimator."""
    if not cls_name:
        return None
    if cls_name in FALLBACK_PARAM_GRIDS:
        return FALLBACK_PARAM_GRIDS[cls_name]
    # Provide gentle defaults for nearby families
    if 'RandomForest' in cls_name:
        if cls_type == 'regressor':
            return FALLBACK_PARAM_GRIDS['RandomForestRegressor']
        return FALLBACK_PARAM_GRIDS['RandomForestClassifier']
    if 'GradientBoosting' in cls_name:
        return {'max_depth': [3, None], 'learning_rate': [0.05, 0.1]}
    if cls_type == 'classifier' and 'LogisticRegression' in cls_name:
        return FALLBACK_PARAM_GRIDS['LogisticRegression']
    if cls_type == 'classifier' and 'SVC' in cls_name:
        return FALLBACK_PARAM_GRIDS['SVC']
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--with-feature-engineering', action='store_true', help='Include a PCA feature engineering step in the fallback wrapper')
    parser.add_argument('--with-tuning', action='store_true', help='Wrap the model with GridSearchCV fallback')
    parser.add_argument('--with-ensemble', action='store_true', help='Wrap the model with a Voting (or Bagging) ensemble fallback')
    parser.add_argument('--tuning-mode', choices=['grid', 'random'], default='grid', help='Which tuning strategy to use for fallback GridSearchCV')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('Source not found:', src)
        return 1

    # Use utf-8-sig to be resilient to BOM in generated files
    text = src.read_text(encoding='utf-8-sig')
    # Try to extract build_full_pipeline function first and create a wrapper that only contains the function
    build_fn_src, import_lines, lines, func_defs = extract_build_fn_and_imports(text)
    disallowed_tokens = ('generated_pipelines',)
    if build_fn_src is not None:
        def _contains_disallowed(text: str) -> bool:
            return any(token in text for token in disallowed_tokens)
        if _contains_disallowed(build_fn_src) or any(_contains_disallowed(line) for line in import_lines):
            # Ignore extracted function if it references unavailable modules (e.g., generated_pipelines)
            build_fn_src = None
            import_lines = []
    cls, cls_type = detect_estimator_class(text)
    if cls is None:
        print('Could not detect known estimator class in source; defaulting to RandomForestRegressor')
        cls = 'RandomForestRegressor'
        cls_type = 'regressor'

    # Build wrapper content
    global_imports = [
        'from sklearn.pipeline import Pipeline',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.impute import SimpleImputer',
    ]
    heavy_imports_in_fn = []
    
    # Wrap heavy library imports in try/except

    # import appropriate estimator
    if cls_type == 'regressor':
        if cls == 'LGBMRegressor':
            heavy_imports_in_fn.append("try:\n    import lightgbm as lgb\nexcept ImportError as e:\n    raise ImportError('lightgbm not installed. Install it with: pip install lightgbm')")
        elif cls == 'CatBoostRegressor':
            heavy_imports_in_fn.append("try:\n    from catboost import CatBoostRegressor\nexcept ImportError as e:\n    raise ImportError('catboost not installed. Install it with: pip install catboost')")
        elif cls == 'XGBRegressor':
            heavy_imports_in_fn.append("try:\n    from xgboost import XGBRegressor\nexcept ImportError as e:\n    raise ImportError('xgboost not installed. Install it with: pip install xgboost')")
        elif cls == 'HistGradientBoostingRegressor':
            global_imports.append('from sklearn.ensemble import HistGradientBoostingRegressor')
        else:
            global_imports.append(f'from sklearn.ensemble import {cls}')
    else:
        if cls == 'HistGradientBoostingClassifier':
            global_imports.append('from sklearn.ensemble import HistGradientBoostingClassifier')
        elif cls == 'XGBClassifier':
            heavy_imports_in_fn.append("try:\n    from xgboost import XGBClassifier\nexcept ImportError as e:\n    raise ImportError('xgboost not installed. Install it with: pip install xgboost')")
        elif cls == 'CatBoostClassifier':
            heavy_imports_in_fn.append("try:\n    from catboost import CatBoostClassifier\nexcept ImportError as e:\n    raise ImportError('catboost not installed. Install it with: pip install catboost')")
        elif cls == 'LogisticRegression':
            global_imports.append('from sklearn.linear_model import LogisticRegression')
        elif cls == 'SVC':
            global_imports.append('from sklearn.svm import SVC')
        else:
            global_imports.append(f'from sklearn.ensemble import {cls}')
    content = []
    content.extend(global_imports)

    # If we successfully extracted build_full_pipeline, use it to produce a safe wrapper
    if build_fn_src is not None:
        # Place extracted imports inside the function to avoid import-time side effects
        fn_lines = build_fn_src.splitlines()
        # Build function header and insert the import lines inside the function
        new_fn = []
        header = fn_lines[0]
        new_fn.append(header)
        # Insert import lines at the top of the function body with indentation
        for im in import_lines:
            new_fn.append('    ' + im.strip())
        # Insert heavy imports at the top of the function body as well
        for im in heavy_imports_in_fn:
            indented = ('    ' + im.replace('\n', '\n' + '    ')).rstrip('\n')
            new_fn.append(indented)
        # Append the original function body (skip the header)
        for line in fn_lines[1:]:
            # If the line is a docstring triple quote start inside function, keep it
            # Replace literal `random_state=42` with the function parameter `random_state` so the wrapper parameter is used
            new_fn.append(line.replace('random_state=42', 'random_state=random_state'))
        # Now detect helper function references inside build_fn and collect those funcs
        helper_funcs_included = set()
        try:
            build_tree = ast.parse(build_fn_src)
            class HelperNameVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.names = set()
                def visit_Name(self, node: ast.Name):
                    self.names.add(node.id)
                    self.generic_visit(node)
            hv = HelperNameVisitor()
            hv.visit(build_tree)
            # Collect names that correspond to top-level function defs
            for name in hv.names:
                if name in func_defs and name != 'build_full_pipeline':
                    helper_funcs_included.add(name)
        except Exception:
            helper_funcs_included = set()

        # Append helper function source code inside the module but before the build function in the wrapper
        helper_src_lines = []
        for name in helper_funcs_included:
            helper_src_lines.append(func_defs.get(name))
        if helper_src_lines:
            content.extend(helper_src_lines)

        # If no imports were present in the original, add a safe preprocessor import inside function
        # as a fallback; we add preprocessor components already globally
        content.append('\n'.join(new_fn))
        # We successfully used the extracted function; skip fallback model creation
        # Ensure the parent directory exists
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(content), encoding='utf-8')
        print('Wrote auto pipeline wrapper (extracted build_full_pipeline):', cls, '->', out)
        return 0
    else:
        content.append('\ndef build_full_pipeline(random_state=42):')
        # Always include a basic preprocessor
        content.append('    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])')
        # Optional feature engineering
        content.append('    feature_engineering = None')
        if args.with_feature_engineering:
            content.append('    try:')
            content.append('        from sklearn.decomposition import PCA')
            content.append('    except Exception:')
            content.append('        PCA = None')
            content.append('    if PCA is not None:')
            content.append('        feature_engineering = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])')
        # Attempt to detect a pipeline in the source and reconstruct
        pipeline_steps = extract_pipeline_steps(text)
        kwlist = 'random_state=random_state'
        est_name = cls
        if pipeline_steps:
            # Use the last step (assumed to be model) for estimator
            last_step = pipeline_steps[-1]
            if last_step.get('estimator_name'):
                est_name = last_step['estimator_name']
            if last_step.get('kwargs_str'):
                kwlist = f"random_state=random_state, {last_step['kwargs_str']}"
        # Add heavy imports inside the function to avoid import-time side effects
        for im in heavy_imports_in_fn:
            indented = ('    ' + im.replace('\n', '\n' + '    ')).rstrip('\n')
            content.append(indented)
    if cls_type == 'regressor':
        if cls == 'LGBMRegressor':
            content.append('    model = lgb.LGBMRegressor(random_state=random_state)')
        elif cls == 'CatBoostRegressor':
            content.append('    model = CatBoostRegressor(random_state=random_state, verbose=0)')
        elif cls == 'HistGradientBoostingRegressor':
            content.append(f'    model = {cls}(random_state=random_state)')
        else:
            content.append(f'    model = {est_name}({kwlist})')
    else:
        if cls == 'HistGradientBoostingClassifier':
            content.append(f'    model = {cls}(random_state=random_state)')
        elif cls == 'XGBClassifier':
            content.append(f'    model = XGBClassifier()')
        elif cls == 'CatBoostClassifier':
            content.append('    model = CatBoostClassifier(verbose=0)')
        else:
            content.append(f'    model = {est_name}({kwlist})')
    # Optionally wrap model with tuning
    if args.with_tuning:
        fallback_grid = get_fallback_param_grid(cls, cls_type)
        fallback_grid_literal = repr(fallback_grid) if fallback_grid is not None else None
        content.append('    param_grid = None')
        content.append('    try:')
        content.append('        from sklearn.model_selection import GridSearchCV')
        content.append('    except Exception:')
        content.append('        GridSearchCV = None')
        if fallback_grid_literal is not None:
            content.append('    if GridSearchCV is not None:')
            content.append(f'        param_grid = {fallback_grid_literal}')
            content.append('        model = GridSearchCV(model, param_grid=param_grid, cv=2, n_jobs=1)')
        else:
            content.append('    # Grid search disabled: no safe param grid for this estimator')
    # Optionally wrap with ensemble
    if args.with_ensemble:
        content.append('    try:')
        content.append('        from sklearn.ensemble import VotingRegressor, VotingClassifier')
        content.append('    except Exception:')
        content.append('        VotingRegressor = None; VotingClassifier = None')
        content.append('    if VotingRegressor is not None or VotingClassifier is not None:')
        if cls_type == 'regressor':
            content.append('        from sklearn.linear_model import LinearRegression')
            content.append('        ensemble = VotingRegressor([("m1", model), ("m2", LinearRegression())])')
            content.append('        model = ensemble')
        else:
            content.append('        from sklearn.linear_model import LogisticRegression')
            content.append('        ensemble = VotingClassifier([("m1", model), ("m2", LogisticRegression())], voting="soft")')
            content.append('        model = ensemble')

    # Build pipeline including feature_engineering if present
    content.append('    steps = [("preprocessor", preprocessor)]')
    content.append('    if feature_engineering is not None:')
    content.append('        steps.append(("feature_engineering", feature_engineering))')
    content.append('    steps.append(("model", model))')
    content.append('    pipeline = Pipeline(steps)')
    content.append('    return pipeline')

    # Ensure the parent directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(content), encoding='utf-8')
    print('Wrote auto pipeline wrapper with estimator:', cls, '->', out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
