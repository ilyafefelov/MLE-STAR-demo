from importlib import util
spec = util.spec_from_file_location('mod', 'model_comparison_results/gemini_live_flash_lite_iris.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
p = mod.build_full_pipeline(random_state=42, deterministic=True)
print('n_jobs present in params:', 'n_jobs' in p.get_params())
n_jobs_params = {k: v for k, v in p.get_params().items() if 'n_jobs' in k}
print('n_jobs params:', n_jobs_params)
