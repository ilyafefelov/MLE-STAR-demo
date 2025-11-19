import runpy
import os, sys

# Ensure repo root on sys.path for imports
sys.path.insert(0, os.getcwd())

ns = runpy.run_path('model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py')
build_full = ns['build_full_pipeline']

from src.mle_star_ablation.mle_star_generated_pipeline import set_full_pipeline_callable, build_pipeline, inspect_pipeline
from src.mle_star_ablation.config import AblationConfig

set_full_pipeline_callable(build_full)

cfg = AblationConfig(name='no_scaling', use_scaling=False, use_feature_engineering=True, use_hyperparam_tuning=True, use_ensembling=True)
pipe = build_pipeline(cfg, deterministic=True, random_state=42)
print(inspect_pipeline(pipe))
