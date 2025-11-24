#!/usr/bin/env python
"""
Utility to inspect pipeline structure for different ablation configurations for debugging.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mle_star_ablation.config import get_standard_configs
from src.mle_star_ablation import mle_star_generated_pipeline as mgp


def import_and_register_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cand_names = ['build_full_pipeline', 'create_model_pipeline', 'create_pipeline']
    for name in cand_names:
        if hasattr(mod, name):
            mgp.set_full_pipeline_callable(getattr(mod, name))
            print(f"Registered {name} from {path}")
            return True
    print("No builder found in the wrapper")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrapper', type=str, required=True)
    parser.add_argument('--deterministic', action='store_true', help='Build pipeline in deterministic mode to set n_jobs=1')
    args = parser.parse_args()
    wrapper = Path(args.wrapper)
    if not wrapper.exists():
        print(f"Wrapper not found: {wrapper}")
        return
    if not import_and_register_wrapper(wrapper):
        return
    configs = get_standard_configs()
    for cfg in configs:
        print('\n' + '='*80)
        print(f"Config: {cfg.name}")
        try:
            pipe = mgp.build_pipeline(cfg, random_state=42, deterministic=args.deterministic)
            info = mgp.inspect_pipeline(pipe)
            print(f"Steps: {info['steps']}")
            print(f"Has scaler: {info['has_scaler']}")
            print(f"Has feature engineering: {info['has_feature_engineering']}")
            print(f"Model type: {info['model_type']}")
            # Report any n_jobs values present
            try:
                def _collect_n_jobs(est):
                    n_jobs_vals = []
                    try:
                        params = est.get_params()
                        if 'n_jobs' in params:
                            n_jobs_vals.append(params['n_jobs'])
                    except Exception:
                        pass
                    if hasattr(est, 'steps'):
                        for _, nested in est.steps:
                            n_jobs_vals.extend(_collect_n_jobs(nested))
                    if hasattr(est, 'transformers'):
                        for _, transformer, _ in est.transformers:
                            n_jobs_vals.extend(_collect_n_jobs(transformer))
                    return n_jobs_vals
                n_jobs_vals = _collect_n_jobs(pipe)
                print(f"n_jobs values found: {n_jobs_vals}")
            except Exception:
                pass
        except Exception as e:
            print(f"Error building pipeline for {cfg.name}: {e}")


if __name__ == '__main__':
    main()
