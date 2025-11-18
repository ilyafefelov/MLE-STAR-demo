
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('gemini_live_iris', r'D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\gemini_live_iris.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['gemini_live_iris'] = mod
spec.loader.exec_module(mod)

builder = getattr(mod, 'build_full_pipeline', None) or getattr(mod, 'create_model_pipeline', None) or getattr(mod, 'create_pipeline', None)

def build_full_pipeline(*args, **kwargs):
    if builder is None:
        raise RuntimeError('Could not find builder function in generated module')
    p = builder(*args, **kwargs)
    if isinstance(p, tuple):
        for el in p:
            if hasattr(el, 'fit'):
                return el
        return p[0]
    return p
