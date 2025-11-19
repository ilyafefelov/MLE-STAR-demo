# Used Pipelines in Experiments

Generated: 2025-11-19

This document lists the pipeline wrapper files used by the experiment manifest `configs/experiment_suite.yaml`.
Each pipeline wrapper defines a `build_full_pipeline()` function used by the ablation runner.

| Experiment Name | Dataset | Task Type | Pipeline file | Notes |
|---|---|---|---|---|
| `cls_breast_cancer` | `breast_cancer` | classification | `model_comparison_results/gemini_2.5_flash_lite_breast_cancer_pipeline_wrapper.py` | Flash-lite wrapper for breast cancer |
| `cls_wine` | `wine` | classification | `model_comparison_results/gemini_2.5_flash_lite_wine_pipeline_wrapper.py` | Flash-lite wrapper for wine |
| `cls_digits` | `digits` | classification | `model_comparison_results/gemini_2.5_flash_lite_digits_pipeline_wrapper.py` | Flash-lite wrapper for digits |
| `cls_iris` | `iris` | classification | `model_comparison_results/gemini_2.5_flash_lite_iris_pipeline_wrapper.py` | Flash-lite wrapper for iris |
| `cls_synthetic_balanced` | `synthetic_classification` | classification | `model_comparison_results/gemini_2.5_flash_lite_digits_pipeline_wrapper.py` | Reuses flash-lite digits wrapper for synthetic benchmark |
| `reg_california` | `california-housing-prices` | regression | `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py` | Live/production-style wrapper (imputer + scaler, PCA, LGBM + ensemble) |
| `reg_diabetes` | `diabetes` | regression | `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py` | Reuses California wrapper for diabetes benchmark |
| `reg_synth_easy` | `synthetic_regression_easy` | regression | `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py` | Reuses California wrapper for synthetic regression |
| `reg_synth_medium` | `synthetic_regression_medium` | regression | `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py` | Reuses California wrapper for synthetic regression |
| `reg_synth_nonlinear` | `synthetic_regression_nonlinear` | regression | `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py` | Reuses California wrapper for Friedman-style nonlinear regression |

How to inspect a pipeline quickly

- Open the pipeline file and look for `build_full_pipeline()` (e.g. `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py`).
- Or run the ad-hoc inspector script used during debugging:

```powershell
python scripts/tmp_inspect_no_scaling.py
```

(That script registers the external pipeline builder with the ablation adapter and prints pipeline structure such as steps, presence of scaler/imputer, and ensemble type.)

Notes

- Several experiments reuse the same wrapper (`gemini_live_california-housing-prices_pipeline_wrapper.py`) for different regression benchmarks—inspect that file to see the full preprocessing and ensemble setup.
- Pipeline wrappers are expected to define `build_full_pipeline()` and are registered at runtime by `scripts/run_ablation.py` via `mgp.set_full_pipeline_callable(...)`.

If you want, I can also:
- Add inline summaries of the pipeline steps (imputer, scaler, PCA, estimator, tuning, ensemble) for each file by parsing them programmatically.
- Produce per-wrapper markdown fragments showing the actual sklearn Pipeline step names.
