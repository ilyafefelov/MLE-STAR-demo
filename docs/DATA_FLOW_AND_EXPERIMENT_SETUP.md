# Data Flow and Experimental Setup (MLE-STAR Demo)

Generated: 2025-11-19

This document explains how datasets, generated wrapper code, and the ablation experiment runner work together. It includes a high-level visual flow, component responsibilities, ablation mechanics, practical commands, and where results are saved.

**High-level flow**

Mermaid flowchart (works in renderers that support Mermaid):

```mermaid
flowchart LR
  subgraph SU[Experiment Orchestration]
    direction LR
    SU1[run_experiment_suite.py\n(manifest)]
    SU2[run_ablation.py\n(ablation loop)]
    SU1 -->|schedules| SU2
  end

  subgraph DS[Data & Loader]
    direction TB
    D1[DatasetLoader]\n(Dataset fetch + train/test split)
  end

  subgraph WR[Wrapper & Adapter]
    direction LR
    W1[Wrapper file\n(model_comparison_results/*_pipeline_wrapper.py)]
    A1[Adapter\n(src/mle_star_ablation/mle_star_generated_pipeline.py)]
    W1 -->|registers| A1
  end

  subgraph PL[Pipeline Execution]
    direction TB
    P1[build_full_pipeline() (wrapper)]
    P2[build_pipeline(config) (adapter)]
    P3[Sklearn Pipeline\n(fit / predict)]
    P1 --> P2 --> P3
  end

  SU2 ---> D1
  SU2 ---> W1
  D1 --> P2
  A1 --> P2
  P3 --> M1[Metrics/Analytics & Visuals]
  M1 --> R1[results/<experiment>/* (CSV, PNG, TXT)]

  classDef orange fill:#FFF4D6,stroke:#C07800
  class SU orange
  class WR orange
```

Rendered PNG (for environments without Mermaid support):

![High-level experiment flow](./diagrams/high_level_flow.png)

**Key files and components**

- `scripts/run_experiment_suite.py` — orchestrates multiple experiments (manifest `configs/experiment_suite.yaml`) and calls `scripts/run_ablation.py` per experiment.
- `scripts/run_ablation.py` — loads dataset, registers the wrapper pipeline, generates ablation `AblationConfig`s, runs repeated fits/predicts for the pipeline variants, and saves detailed results, summary CSVs, statistical reports, and plots.
- `model_comparison_results/*_pipeline_wrapper.py` — pipeline wrappers created by MLE-STAR (Gemini) or hand-crafted wrappers. They expose (or define) `build_full_pipeline(random_state=...)` returning a sklearn Pipeline or estimator.
- `src/mle_star_ablation/mle_star_generated_pipeline.py` — adapter that:
  - Accepts `build_full_pipeline` as callable via `set_full_pipeline_callable()`.
  - Normalizes step names (preprocessor, feature_engineering, model).
  - Applies ablation toggles (no_scaling, no_feature_engineering, no_tuning, no_ensemble, minimal).
  - Provides `inspect_pipeline()` and `print_pipeline_structure()` utilities.
- `src/mle_star_ablation/config.py` — defines `AblationConfig` and `get_standard_configs()`.
- `src/mle_star_ablation/datasets.py` — `DatasetLoader` used to fetch and prepare datasets.
- `results/` — output directory with per-experiment folders, summary stats, pairwise comparisons, statistical report files, and PNGs.

**Ablation examples (Mermaid)**

```mermaid
flowchart LR
  subgraph Full[Full pipeline]
    direction LR
    FI[Imputer] --> FS[Scaler]
    FS --> FP[PCA]
    FP --> FG[GridSearchCV(LGBM)]
    FG --> FE[VotingRegressor]
  end

  subgraph NoScaling[No scaling]
    direction LR
    NI[Imputer] --> NX[Identity (no scaler)]
    NX --> NP[PCA]
    NP --> NG[GridSearchCV(LGBM)]
    NG --> NE[VotingRegressor]
  end

  subgraph NoTuning[No tuning]
    direction LR
    TI[Imputer] --> TS[Scaler]
    TS --> TP[PCA]
    TP --> TE[Estimator (no GridSearch)]
    TE --> TE2[VotingRegressor]
  end

  subgraph NoEnsemble[No ensemble]
    direction LR
    EI[Imputer] --> ES[Scaler]
    ES --> EP[PCA]
    EP --> EG[GridSearchCV(LGBM)]
    EG --> EF[Fallback estimator]  
  end

  style Full fill:#eaf6ff,stroke:#2b7ca6
  style NoScaling fill:#fff9db,stroke:#c78d00
  style NoTuning fill:#f3fff1,stroke:#2ea02e
  style NoEnsemble fill:#fff0f0,stroke:#c02e2e
```

Rendered PNG (fallback):

![Ablation examples](./diagrams/ablation_examples_refined.png)

**How data isn't broken when steps are turned off**

- The adapter never blindly removes a preprocessing container: it either removes only a step (e.g., scaler) inside a nested `Pipeline`, or replaces removed preprocessor/FE components with identity `FunctionTransformer` or keeps the imputer. This preserves data shape and avoids NaNs.
- When `use_scaling=False`: the adapter removes the scaler inside the preprocessor but retains the imputer (so missing values are still handled).
- When `use_feature_engineering=False`: the adapter removes or replaces FE step with identity preserving shape.
- When `use_hyperparam_tuning=False`: the adapter unwraps `GridSearchCV` / `RandomizedSearchCV` to the inner estimator template (`estimator` attribute).
- When `use_ensembling=False`: the adapter attempts to pick a single fallback estimator from the ensemble (e.g., first estimator from `VotingRegressor`) or the final estimator from `StackingClassifier`.
- These operations keep the pipeline consistent and prevent runtime shape/feature errors.

**A good example run**

1. Register wrapper and run ablation from script:

```powershell
# Run single dataset with manifest or pipeline file
python scripts/run_ablation.py --dataset california-housing-prices --pipeline-file model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py --n-runs 20 --output-dir results/reg_california
```

2. Or run an experiment manifest subset:

```powershell
python scripts/run_experiment_suite.py --config configs/experiment_suite.yaml --only reg_california --execute
```

**Where results appear**

- Top-level `results/<experiment_name>/...` containing:
  - `detailed_results_*.csv` — per-run metrics (R2, accuracy, rmse, train_time, etc.)
  - `summary_statistics_*.csv` — aggregated mean/std/CI for each config
  - `statistical_report_*.txt` — ANOVA and pairwise comparisons results
  - `pairwise_comparisons_*.csv` — detailed pairwise t-tests among configs
  - PNGs for plots produced by `src/mle_star_ablation/viz`.

**Sanity tests**

- `scripts/validate_wrappers_ablation.py` — collects step names across wrappers and variant types.
- `scripts/run_wrapper_sanity_checks.py` — builds each adapter variant and attempts `fit` + `predict` on a small subset to ensure there are no runtime errors for `build_pipeline()`-based variants.

Example run to assert all wrappers pass a basic fit/predict check:

```powershell
python scripts/run_wrapper_sanity_checks.py --config configs/experiment_suite.yaml --wrappers model_comparison_results --out docs/WRAPPER_SANITY_RESULTS
```

**Diagnosing anomalies**

If a pipeline variant produces unexpectedly poor metrics (e.g., `no_scaling` yields negative R2 while `minimal` is good):
- Check `docs/PIPELINE_ABLATION_SUMMARY.md` to confirm whether the adapter is actually preserving the imputer and only removing the scaler.
- Check `docs/WRAPPER_SANITY_RESULTS.md` to confirm the `fit/predict` success for the variant.
- Inspect the wrapper itself in `docs/ALL_PIPELINES_FULL.md` to see the original GeminI-generated pipeline (the verbatim wrapper code is captured there).
- If any component is missing (e.g., the wrapper removes the preprocessor entirely), fix the wrapper or adjust adapter logic.

**How reproducibility is handled**

- For each run: `scripts/run_ablation.py` uses `random_state` and optionally `--deterministic` flags to set seeds and attempt deterministic BLAS. The adapter also tries to set nested `random_state` and `n_jobs` for determinism.
- Results include the config, the run random_state, and all metrics; the wrapper source is included in `docs/ALL_PIPELINES_FULL.md` and can be saved in the `results` folder for reproducibility.

**Troubleshooting**

- If the `no_scaling` variant fails with NaNs: check the adapter or wrapper for accidental full preprocessor removal and re-run `scripts/run_wrapper_sanity_checks.py` for targeted diagnostics.
- If `no_ensemble` returns a list instead of estimator: the adapter uses `_unwrap_ensemble()`; verify wrapper ensemble types and fallback logic.
- If GridSearchCV is still wrapped under `no_tuning`: ensure adapter is correctly unwrapping by checking `inspect_pipeline()` output.

**Recommended next steps / CI additions**

- Add `scripts/run_wrapper_sanity_checks.py` to CI to prevent regressions in wrapper/adapter behavior.
- Save the wrapper source used for each run to `results/<experiment_name>/wrapper_source.py` for future audits.
- Optionally, add small unit tests to assert that `no_scaling` preserves imputer and that `no_tuning` unwraps `GridSearchCV` for a couple of commonly used wrappers.

---

If you'd like, I can also:
- Generate a PNG/diagram file (SVG, PNG) in `docs/` using the Mermaid diagrams or an additional image library with a higher-quality visual.
- Insert a small test to automatically copy the wrapper file into the `results` directory during an experiment run for reproducibility.

Which next improvement would you prefer? (Add to CI, create PNG diagram, copy wrapper into `results` automatically)