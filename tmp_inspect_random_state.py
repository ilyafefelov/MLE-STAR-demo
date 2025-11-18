from importlib import util
from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs
spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_iris.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Register the builder
mgp.set_full_pipeline_callable(mod.build_full_pipeline)
# Create config
configs = get_standard_configs()
config = configs[0]
# Build pipeline in deterministic mode with random_state=99
p = mgp.build_pipeline(config, random_state=99, deterministic=True)

print('Pipeline type:', type(p))
# Inspect steps and random_state values
for name, step in p.steps:
    print('STEP:', name, type(step))
    # check n_jobs and random_state
    print('  n_jobs ->', getattr(step, 'n_jobs', 'No n_jobs'))
    print('  random_state ->', getattr(step, 'random_state', 'No random_state'))
    # nested steps if pipeline
    if hasattr(step, 'steps'):
        for sub_name, sub in step.steps:
            print('   nested:', sub_name, type(sub), 'n_jobs->', getattr(sub, 'n_jobs', 'No n_jobs'), 'random_state->', getattr(sub, 'random_state', 'No rs'))
    # Handle ColumnTransformer
    if hasattr(step, 'transformers'):
        for trans in step.transformers:
            name2, trans_obj, cols = trans
            print('   column transformer part:', name2, type(trans_obj), 'n_jobs->', getattr(trans_obj, 'n_jobs', 'No n_jobs'), 'random_state->', getattr(trans_obj, 'random_state', 'No rs'))
