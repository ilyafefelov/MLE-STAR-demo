from importlib import util
from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs
from sklearn.pipeline import Pipeline

spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_digits.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
mgp.set_full_pipeline_callable(mod.build_full_pipeline)
configs = get_standard_configs()
config = configs[0]
p = mgp.build_pipeline(config, random_state=42, deterministic=True)

print('Pipeline steps:')

def inspect_obj(obj, prefix=''):
    t = type(obj)
    s = f"{prefix}{t}"
    nj = getattr(obj, 'n_jobs', 'No n_jobs attr')
    print(s, 'n_jobs ->', nj)
    if isinstance(obj, Pipeline):
        for name, step in obj.steps:
            inspect_obj(step, prefix + name + ' -> ')
    # estimators for ensembles
    if hasattr(obj, 'estimators'):
        for est in getattr(obj, 'estimators'):
            inspect_obj(est, prefix + 'estimators-> ')
    if hasattr(obj, 'estimator') and obj.estimator is not None:
        inspect_obj(obj.estimator, prefix + 'estimator-> ')

inspect_obj(p)
