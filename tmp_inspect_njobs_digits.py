from importlib import util
from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs
spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_digits.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
mgp.set_full_pipeline_callable(mod.build_full_pipeline)
configs = get_standard_configs()
config = configs[0]
p = mgp.build_pipeline(config, random_state=42, deterministic=True)
print('Pipeline type:', type(p))
print('n_jobs present in params:', any(['n_jobs' in k for k in p.get_params().keys()]))
n_jobs_params = {k: v for k, v in p.get_params().items() if 'n_jobs' in k}
print('n_jobs params:', n_jobs_params)
for name, step in p.steps:
    print(name, type(step), getattr(step, 'n_jobs', 'No n_jobs attr'))
    if hasattr(step, 'estimators'):
        for est in step.estimators:
            print('  ensemble est:', type(est), getattr(est, 'n_jobs', 'No n_jobs attr'))
    if hasattr(step, 'estimator'):
        est = step.estimator
        print('  estimator:', type(est), getattr(est, 'n_jobs', 'No n_jobs attr'))
