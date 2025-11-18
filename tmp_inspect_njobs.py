from importlib import util
from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs
spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_iris.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
mgp.set_full_pipeline_callable(mod.build_full_pipeline)
configs = get_standard_configs()
config = configs[0]
p = mgp.build_pipeline(config, random_state=42, deterministic=True)
print('Pipeline steps:')
for name, step in p.steps:
    print(name, type(step), getattr(step, 'n_jobs', 'No n_jobs attr'))
    if hasattr(step, 'named_steps'):
        for subname, substep in step.named_steps.items():
            print('   substep', subname, type(substep), getattr(substep, 'n_jobs', 'No n_jobs attr'))
    if hasattr(step, 'estimator'):
        est = step.estimator
        print('  estimator:', type(est), getattr(est, 'n_jobs', 'No n_jobs attr'))
