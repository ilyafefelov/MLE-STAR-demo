from importlib import util
from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs
spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_iris.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Register the external builder
mgp.set_full_pipeline_callable(mod.build_full_pipeline)
# Create a sample config
configs = get_standard_configs()
config = configs[0]
p = mgp.build_pipeline(config, random_state=42, deterministic=True)
print('Pipeline type:', type(p))
print('n_jobs present in params:', any(['n_jobs' in k for k in p.get_params().keys()]))
print('n_jobs param values:')
for k, v in p.get_params().items():
    if 'n_jobs' in k:
        print(k, v)
