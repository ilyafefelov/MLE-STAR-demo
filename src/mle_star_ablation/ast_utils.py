"""
AST utilities to safely inject and propagate `random_state` into generated build_full_pipeline functions.

Functions:
- inject_random_state_into_build_fn: Returns new source with `random_state` arg in signature
  and all internal `random_state=<const>` keywords replaced with `random_state` name.
"""
from __future__ import annotations

import ast
from typing import Optional


class RandomStateInjector(ast.NodeTransformer):
    """NodeTransformer to replace keyword=literal with keyword=random_state Name.

    Example: PCA(..., random_state=42) -> PCA(..., random_state=random_state)
    Only replaces inside the function body.
    """

    def visit_Call(self, node: ast.Call):
        # Process keywords first
        for kw in node.keywords:
            if kw.arg == 'random_state' and isinstance(kw.value, (ast.Num, ast.Constant)):
                # Replace with Name('random_state')
                kw.value = ast.Name(id='random_state', ctx=ast.Load())
        # Recurse
        self.generic_visit(node)
        return node


class EstimatorRandomStateAdder(ast.NodeTransformer):
    """Add random_state keyword in constructor calls for known estimator names when missing.
    This aims to ensure nested estimators use the function's random_state param.
    """
    def __init__(self, estimator_names: Optional[list] = None):
        # Default set of estimators that commonly accept random_state
        self.estimator_names = estimator_names or [
            'RandomForestClassifier', 'RandomForestRegressor', 'PCA', 'LogisticRegression',
            'GradientBoostingClassifier', 'MLPClassifier', 'SVC', 'KNeighborsClassifier',
            'DecisionTreeClassifier', 'DecisionTreeRegressor', 'KNeighborsRegressor'
        ]

    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name in self.estimator_names:
            # If random_state is missing, add random_state reuse
            if not any(kw.arg == 'random_state' for kw in node.keywords):
                node.keywords.append(ast.keyword(arg='random_state', value=ast.Name(id='random_state', ctx=ast.Load())))
        self.generic_visit(node)
        return node


class NJobsInjector(ast.NodeTransformer):
    """Add or replace n_jobs kwargs on estimator constructor calls with a given value.
    This ensures generated code uses `n_jobs=1` when required for deterministic runs.
    """
    def __init__(self, n_jobs_val: int = 1, names: Optional[list] = None):
        self.n_jobs_val = n_jobs_val
        # Optional list of constructor names to target; if None, apply to all
        self.names = names

    def visit_Call(self, node: ast.Call):
        # If names filter is present and the func name is not in names, skip
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        # Only proceed if name matches or no names filter
        if self.names is None or (func_name and func_name in self.names):
            found = False
            for kw in node.keywords:
                if kw.arg == 'n_jobs':
                    found = True
                    kw.value = ast.Constant(value=self.n_jobs_val)
            if not found:
                # Append a keyword
                node.keywords.append(ast.keyword(arg='n_jobs', value=ast.Constant(value=self.n_jobs_val)))
        self.generic_visit(node)
        return node


def ensure_random_state_arg(fn: ast.FunctionDef) -> bool:
    """Add `random_state: int = 42` argument to FunctionDef if missing.

    Returns True if modified.
    """
    arg_names = [a.arg for a in fn.args.args]
    if 'random_state' in arg_names:
        return False
    # Create new arg with default
    import sys
    # Build arg object
    new_arg = ast.arg(arg='random_state', annotation=ast.Name(id='int', ctx=ast.Load()))
    # Prepend to args
    fn.args.args.insert(0, new_arg)
    # Insert default value
    # Defaults are aligned from the end; ensure defaults length matches tail args
    # For simplicity we add default to defaults list (works for inserting at beginning)
    const = ast.Constant(value=42)
    fn.args.defaults.insert(0, const)
    return True


def inject_random_state_into_build_fn(source: str) -> str:
    """Given a textual Python source that contains def build_full_pipeline(...):
    return a modified source string where the function signature includes
    `random_state: int = 42` (if absent) and any keyword kwargs `random_state=<const>`
    within the function body are replaced with a Name('random_state').

    This uses AST-based transformations and returns the new source code.
    """
    tree = ast.parse(source)
    modified = False

    class FnTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            nonlocal modified
            if node.name == 'build_full_pipeline':
                if ensure_random_state_arg(node):
                    modified = True
                injector = RandomStateInjector()
                injector.visit(node)
            return node

    transformer = FnTransformer()
    transformer.visit(tree)

    if not modified:
        # Still attempt to replace internal calls even if arg existed
        injector = RandomStateInjector()
        injector.visit(tree)
    # Regardless of the initial modification, always attempt to add estimator random_state and n_jobs
    estimator_rs_injector = EstimatorRandomStateAdder()
    estimator_rs_injector.visit(tree)
    njobs_injector = NJobsInjector(n_jobs_val=1, names=None)
    njobs_injector.visit(tree)
    # Also inject n_jobs=1 by default for classifiers/estimators often parallelized (GridSearchCV, RandomForest, etc.)
    njobs_injector = NJobsInjector(n_jobs_val=1, names=['GridSearchCV', 'RandomizedSearchCV', 'RandomForestClassifier', 'RandomForestRegressor', 'RandomizedSearchCV', 'BaggingClassifier'])
    njobs_injector.visit(tree)

    # Unparse AST back to source
    try:
        new_source = ast.unparse(tree)
    except Exception:
        # For older Python without robust unparse, fall back to original
        return source
    return new_source


def inject_random_state_into_file(src_path: str, dst_path: Optional[str] = None) -> str:
    """Helper to read a file, apply injection, and write to destination.
    Returns path to written file (or original file if not modified).
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src = f.read()
    new_src = inject_random_state_into_build_fn(src)
    out_path = dst_path if dst_path else src_path
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_src)
    return out_path


__all__ = [
    'inject_random_state_into_build_fn',
    'inject_random_state_into_file',
]
