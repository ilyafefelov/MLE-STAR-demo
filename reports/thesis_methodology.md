# Methodology — MLE-STAR + Gemini ablation study

## Overview
This study investigates automatically generated ML pipelines using the Gemini API in the context of an MLE-STAR ablation protocol. The goal is to evaluate impact of several pipeline components (scaling, feature engineering, hyperparameter tuning, ensembling) on classification performance across four classical UCI datasets: `breast_cancer`, `wine`, `digits`, `iris`.

We use an automated pipeline generation approach: Gemini-generated Python pipelines and short, human-readable pipeline selection prompts produce a full sklearn `Pipeline` that we subsequently evaluate under controlled ablation settings.

## Datasets and splits
- Datasets: `breast_cancer` (n≈569), `wine` (n≈178), `digits` (n≈1797), `iris` (n≈150).
- Train/test split: by default we use 80/20 (train/test) and fix seeds per run (see reproducibility section).
- Some experiments used a single-run smoke configuration (n_runs=1) to quickly validate pipeline code; core experiments used `n_runs=5` or `n_runs=10` to estimate mean and variance.

## Pipeline variants and ablation protocols
We define a baseline `full` pipeline that includes the following components (where available in a generated pipeline):
- Preprocessing (imputation, scaling) — `StandardScaler` used when `use_scaling` is true.
- Feature engineering (e.g., `PCA`) — optional.
- Hyperparameter tuning with `GridSearchCV` applied to a default model (RandomForestClassifier, SVC, GradientBoosting, or MLP depending on the generated pipeline).
- Ensembling (where available) such as a stacked or voting classifier.

Ablation configurations (explicitly toggled):
- `minimal`: no scaling, no feature engineering, no hyperparameter tuning, no ensembling (baseline simplest model).
- `no_scaling`: remove the scaling step.
- `no_feature_engineering`: remove engineering such as PCA.
- `no_tuning`: skip GridSearchCV hyperparameter search.
- `no_ensemble`: turn off ensemble meta-models.
- `full`: baseline pipeline as generated (with all components).

Each configuration is tested `n_runs` times with different seeds; the mean accuracy, standard deviation, and 95% confidence intervals are saved in `reports/table2_ablation.csv`.

## Gemini variant comparison (Model Comparison / Table 1)
To evaluate the generated pipelines from different Gemini variants, we compared code generation outputs from three pre-defined variants: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`. We collected the following metrics per generated pipeline for Table 1: mean accuracy (validation/test), standard deviation, generation time (seconds), code length (number of tokens/characters), chosen model type (e.g., `RandomForestClassifier`, `SVC`), and pipeline steps. The aggregated model comparison CSV is `reports/table1_model_comparison.csv`.

## Metrics and statistical tests
- Primary metrics: Accuracy (classification accuracy) and F1-score (macro average). Secondary metrics: training time (seconds), code length (for model complexity analysis).
- Summary reported per configuration: mean, standard deviation, minimum & maximum obtained across runs, 95% confidence interval (CI).
- Statistical tests: one-way ANOVA to detect differences across configurations; if significant, we computed pairwise comparisons using t-tests and Bonferroni correction for multiple comparisons. Additionally we report Cohen's d for pairwise effect sizes.

## Reproducibility and execution instructions
The project supports dynamic pipeline registration and variant selection via command line flags.

- To run a full ablation for dataset `wine` with 5 repeats:
```pwsh
& "C:/Program Files/Python313/python.exe" scripts/run_ablation.py --dataset wine --n-runs 5 --output-dir results/wine_n5 --verbose
```

- To run a generated pipeline from a variant file (e.g., `gemini_2.5_pro_wine.py`):
```pwsh
& "C:/Program Files/Python313/python.exe" scripts/run_ablation.py --dataset wine --n-runs 5 --variant pro --output-dir results/wine_pro_n5 --verbose
```

- To re-run all variants across multiple datasets with basic automation, use:
```pwsh
& "C:/Program Files/Python313/python.exe" scripts/run_experiments_variants.py --variants flash_lite,flash,pro --datasets breast_cancer,wine,digits,iris --n-runs 5 --output-dir results
```

- After runs complete, aggregate outputs:
```pwsh
& "C:/Program Files/Python313/python.exe" scripts/aggregate_model_comparison.py --input-dir model_comparison_results --out reports/table1_model_comparison.csv
& "C:/Program Files/Python313/python.exe" scripts/aggregate_results.py --results-dir results --outdir reports
& "C:/Program Files/Python313/python.exe" scripts/clean_reports.py --reports reports
```

## Limitations
- Not all generated pipelines contain every component (e.g., a pipeline may not have a `GridSearchCV` step); in such cases an ablation toggle has no effect and is documented in the final tables.
- `smoke` runs (single repetition) are not robust; they are intended to verify runtime code correctness only.
- The study is limited to classical small datasets; GPU/large-scale training considerations are not addressed.

---

References to the code used for the experiment are saved under `scripts/` and `src/mle_star_ablation/` for the run harness, ablation config definition, metrics, reproducible pipeline registration (`mle_star_generated_pipeline.py`), and viz/stat modules.
