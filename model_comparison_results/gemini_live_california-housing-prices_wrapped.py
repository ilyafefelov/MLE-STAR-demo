
    import importlib.util
    import sys
    from pathlib import Path

    # Load the generated module by path without running fit() automatically
    spec = importlib.util.spec_from_file_location('gemini_live_california-housing-prices', r'D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\gemini_live_california-housing-prices.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['gemini_live_california-housing-prices'] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # If the module executed training code during import, we will not try to import further here
        pass

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
    