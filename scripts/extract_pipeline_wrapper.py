#!/usr/bin/env python
"""
Create a safe wrapper exposing build_full_pipeline() for a training-style script that doesn't expose a builder.

Usage:
  python scripts/extract_pipeline_wrapper.py --src model_comparison_results/gemini_live_iris.py --out model_comparison_results/gemini_live_iris_wrapper_auto.py
"""
import argparse
import ast
from pathlib import Path
from typing import Optional
from textwrap import indent


def _get_source(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


class EstimatorInfo:
    def __init__(self, varname: str, cls_name: str, call_str: str):
        self.varname = varname
        self.cls_name = cls_name
        self.call_str = call_str


def extract_pipeline_code(src_path: Path):
    text = src_path.read_text(encoding='utf-8')
    import_lines = [l.strip() for l in text.splitlines() if l.strip().startswith('import ') or l.strip().startswith('from ')]
    parsed = ast.parse(text)

    estimator_names = set([
        'HistGradientBoostingClassifier', 'RandomForestClassifier', 'RandomForestRegressor', 'LogisticRegression',
        'SVC', 'XGBClassifier', 'GradientBoostingClassifier'
    ])

    found_estimators: dict[str, EstimatorInfo] = {}
    found_preprocessor: Optional[str] = None
    found_pipeline: Optional[str] = None
    found_column_transformers: dict[str, str] = {}
    found_feature_eng: dict[str, str] = {}
    found_gridsearch_var: Optional[str] = None

    for node in parsed.body:
        # Capture top-level Assign nodes
        if isinstance(node, ast.Assign):
            # Right-hand side is a call
            value = node.value
            if isinstance(value, ast.Call):
                # Function can be Name or Attribute
                func = value.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                # Detect Pipeline instantiation
                if func_name == 'Pipeline':
                    found_pipeline = _get_source(text, node)
                # Detect ColumnTransformer/transformers
                if func_name == 'ColumnTransformer':
                    if isinstance(node.targets[0], ast.Name):
                        found_column_transformers[node.targets[0].id] = _get_source(text, node)
                # Detect GridSearchCV
                if func_name == 'GridSearchCV':
                    # record varname
                    if isinstance(node.targets[0], ast.Name):
                        found_gridsearch_var = node.targets[0].id
                # Detect estimator instantiation
                if func_name in estimator_names:
                    if isinstance(node.targets[0], ast.Name):
                        varname = node.targets[0].id
                        call_str = _get_source(text, value)
                        found_estimators[varname] = EstimatorInfo(varname, func_name, call_str)
                # Capture PCA or feature engineering pipelines
                if func_name in ('PCA', 'SelectKBest', 'PolynomialFeatures'):
                    if isinstance(node.targets[0], ast.Name):
                        found_feature_eng[node.targets[0].id] = _get_source(text, node)
            # Also detect simple names referencing preprocessor or pipeline
            if isinstance(value, ast.Name) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'preprocessor':
                    found_preprocessor = _get_source(text, node)
                if node.targets[0].id == 'pipeline':
                    found_pipeline = _get_source(text, node)
        elif isinstance(node, ast.FunctionDef):
            # If the agent provided a builder function, return it
            if node.name in ('build_full_pipeline', 'create_model_pipeline', 'create_pipeline'):
                return import_lines, node.name, _get_source(text, node)

    # If found grid search, try to examine its args to find the inner estimator var or the inline instantiation
    grid_estimator_var = None
    if found_gridsearch_var:
        # find the assign node for gridsearch var
        for node in parsed.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == found_gridsearch_var:
                # Find 'estimator' kw or positional first arg
                call = node.value
                if isinstance(call, ast.Call):
                    # check keywords
                    for kw in call.keywords:
                        if kw.arg == 'estimator':
                            if isinstance(kw.value, ast.Name):
                                grid_estimator_var = kw.value.id
                        # other cases (inline call) we can get source of kw.value
                    # if no kw, look at first arg
                    if not grid_estimator_var and call.args:
                        arg0 = call.args[0]
                        if isinstance(arg0, ast.Name):
                            grid_estimator_var = arg0.id
                break

    # build a preferred estimator to return
    chosen_estimator: Optional[EstimatorInfo] = None
    if grid_estimator_var and grid_estimator_var in found_estimators:
        chosen_estimator = found_estimators[grid_estimator_var]
    elif found_estimators:
        # pick the first one
        chosen_estimator = next(iter(found_estimators.values()))

    # Return a tuple: imports, chosen_var_name or function name, and code_block to create estimator/preprocessor
    if chosen_estimator:
        code_block = ''
        # If preprocessor was found, include the preprocessor assignment snippet
        # preferentially include exact preprocessor assignment
        if found_preprocessor:
            code_block += found_preprocessor + '\n\n'
        else:
            # try to include column transformer if found
            if found_column_transformers:
                for name, src in found_column_transformers.items():
                    code_block += src + '\n\n'
        # include any feature engineering assignments
        for fe in found_feature_eng.values():
            code_block += fe + '\n\n'
        code_block += f"{chosen_estimator.varname} = {chosen_estimator.call_str}\n"
        return import_lines, chosen_estimator.varname, code_block

    # Fallbacks: if pipeline assigned, return pipeline var name and pipeline code
    if found_pipeline:
        return import_lines, 'pipeline', found_pipeline

    # No clear pipeline or estimator found
    return import_lines, None, None


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

    imports, var_name, code = extract_pipeline_code(src)
    if var_name is None or code is None:
        print('Could not detect pipeline definition in source file; aborting')
        return 2

    # Build wrapper content
    content = []
    # Ensure essential imports exist
    essential = [
        'from sklearn.pipeline import Pipeline',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.impute import SimpleImputer',
    ]
    imports_set = set(imports)
    for imp in essential:
        if imp not in imports_set:
            content.append(imp)
    content.extend(imports)
    content.append('\n')
    content.append('def build_full_pipeline(random_state=42):')
    # Insert the code block (indent by 4 spaces)
    for line in code.splitlines():
        content.append('    ' + line)
    content.append('')
    # If pipeline variable is present, return it, else construct a minimal Pipeline using the detected estimator
    content.append('    # Build and return a scikit-learn Pipeline using detected components')
    content.append('    try:')
    content.append('        if "pipeline" in locals():')
    content.append('            return pipeline')
    content.append('    except Exception:')
    content.append('        pass')
    content.append('')
    # Now build an explicit pipeline with preprocessor and model
    content.append('    preprocessor = Pipeline([(\"imputer\", SimpleImputer(strategy=\"mean\")), (\"scaler\", StandardScaler())])')
    # model var: if var_name is not pipeline, we expect that var exists in locals due to code snippet; else default
    content.append(f'    model = {var_name} if "{var_name}" in globals() or "{var_name}" in locals() else None')
    content.append('    if model is None:')
    content.append('        raise RuntimeError("Could not detect model or pipeline from generated script")')
    content.append('    pipeline = Pipeline([(\"preprocessor\", preprocessor), (\"model\", model)])')
    content.append('    return pipeline')
    # If var_name points to grid_search or best estimator, attempt to return estimator object
    # Wrapper always returns a pipeline; the code above builds it using detected 'pipeline' var or model var

    out.write_text('\n'.join(content), encoding='utf-8')
    print('Wrote wrapper to', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
