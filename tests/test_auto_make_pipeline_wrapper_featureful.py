import sys
import importlib
import tempfile
from pathlib import Path
import importlib.util
from textwrap import dedent

import importlib.util
import sys
from pathlib import Path
auto_make_pipeline_wrapper_path = Path('scripts') / 'auto_make_pipeline_wrapper.py'
spec = importlib.util.spec_from_file_location('auto_make_pipeline_wrapper', str(auto_make_pipeline_wrapper_path))
auto_make_pipeline_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_make_pipeline_wrapper)


def write_temp_script(tmpdir: Path, content: str) -> Path:
    p = tmpdir / 'dummy_train.py'
    p.write_text(content, encoding='utf-8')
    return p


def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_generator_with_featureful_flags(tmp_path):
    src = write_temp_script(tmp_path, dedent('''
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        model = RandomForestRegressor(n_estimators=10)
        # simulate train
        def train():
            return model
    '''))
    out = tmp_path / 'out_wrapper.py'
    # build args and call main
    sys_argv = ['auto_make_pipeline_wrapper.py', '--src', str(src), '--out', str(out), '--with-feature-engineering', '--with-tuning', '--with-ensemble']
    old_argv = sys.argv[:]
    try:
        sys.argv[:] = sys_argv
        auto_make_pipeline_wrapper.main()
    finally:
        sys.argv[:] = old_argv
    # Import generated wrapper
    mod = import_module_from_path(out)
    pipe = mod.build_full_pipeline(random_state=42)
    # Expect pipeline steps: preprocessor, feature_engineering (PCA), model
    steps = [name for name, _ in pipe.steps]
    assert any('preprocessor' in s for s in steps)
    assert any('feature_engineering' in s or 'pca' in s for s in steps)
    # Check model exists
    assert any('model' in s for s in steps)
