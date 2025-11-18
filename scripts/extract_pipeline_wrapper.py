#!/usr/bin/env python
"""
Create a safe wrapper exposing build_full_pipeline() for a training-style script that doesn't expose a builder.

Usage:
  python scripts/extract_pipeline_wrapper.py --src model_comparison_results/gemini_live_iris.py --out model_comparison_results/gemini_live_iris_wrapper_auto.py
"""
import argparse
import ast
from pathlib import Path


def extract_pipeline_code(src_path: Path):
    text = src_path.read_text(encoding='utf-8')
    # Collect import lines
    import_lines = [l.strip() for l in text.splitlines() if l.strip().startswith('import ') or l.strip().startswith('from ')]
    # Candidate variable names we might want to extract
    candidates = {'grid_search', 'pipeline', 'hgb_model', 'param_grid', 'clf', 'model', 'best_gbdt_model', 'best_model', 'best_estimator'}

    # Parse the AST and find assignments for candidate vars
    parsed = ast.parse(text)
    assigns = []
    last_var = None
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in candidates:
                    snippet = ast.get_source_segment(text, node)
                    assigns.append(snippet)
                    last_var = target.id
        # Also capture function defs of pipeline builder
        if isinstance(node, ast.FunctionDef):
            if node.name in ('build_full_pipeline', 'create_model_pipeline', 'create_pipeline'):
                # Return the whole function source segment
                assigns.append(ast.get_source_segment(text, node))
                last_var = node.name

    # If we have found any assignment snippets, join them with imports
    # If no assignment found, try to find the first call to GridSearchCV or Pipeline
    if assigns:
        # Remove references to `.best_estimator_` since we won't run fit() and that attribute won't exist
        code_block = '\n\n'.join(assigns)
        code_lines = [l for l in code_block.splitlines() if '.best_estimator_' not in l]
        code_block = '\n'.join(code_lines)
        return import_lines, last_var, code_block
    # fallback: look for 'GridSearchCV(' or 'Pipeline(' occurrences in text
    for token in ('GridSearchCV(', 'Pipeline('):
        idx = text.find(token)
        if idx != -1:
            start = text.rfind('\n', 0, idx) + 1
            # Extract line
            end = text.find('\n', idx)
            if end == -1:
                end = len(text)
            code_block = text[start:end]
            # guess var name
            var_guess = 'grid_search' if 'GridSearchCV' in token else 'pipeline'
            return import_lines, var_guess, code_block
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
    content.extend(imports)
    content.append('\n')
    content.append('def build_full_pipeline(random_state=42):')
    # Insert the code block (indent by 4 spaces)
    for line in code.splitlines():
        content.append('    ' + line)
    # If var_name points to grid_search or best estimator, attempt to return estimator object
    if var_name and ('grid_search' in var_name or 'best' in var_name):
        content.append('    # Return the inner estimator template, not a fitted object')
        content.append('    try:')
        content.append('        if "grid_search" in locals():')
        content.append('            return grid_search.estimator')
        content.append('        if "hgb_model" in locals():')
        content.append('            return hgb_model')
        content.append('        if "clf" in locals():')
        content.append('            return clf')
        content.append('        # No fitted \'best_model\' available; fallback to available unfitted estimator')
        content.append('        raise RuntimeError(\'No fitted best_model available; returning registry-estimator instead\')')
        content.append('    except Exception:')
        content.append('        # Fallback: return the pipeline variable if present')
        content.append('        try:')
        content.append('            return pipeline')
        content.append('        except Exception:')
        content.append('            raise RuntimeError("No pipeline-like variable found")')
    else:
        if var_name:
            content.append('    return ' + var_name)
        else:
            content.append('    raise RuntimeError("No pipeline-like variable found")')

    out.write_text('\n'.join(content), encoding='utf-8')
    print('Wrote wrapper to', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
