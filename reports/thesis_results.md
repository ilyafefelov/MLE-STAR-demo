# Results — Model Comparison & Ablation Study

This document summarizes the key experimental results derived from the generated pipelines and ablation analysis. Full data for the generated models and ablation statistics is saved in `reports/table1_model_comparison.csv` and `reports/table2_ablation.csv` respectively.

## Table 1 — Model Comparison (per dataset)
The best performing pipeline variant per dataset (aggregated across 5 runs where available) is:

- breast_cancer: `gemini-2.5-flash` — mean accuracy 0.9508 (std 0.0153) using a GradientBoostingClassifier. See `reports/table1_model_comparison.csv` for details.
- wine: `gemini-2.5-pro` — mean accuracy 0.9719 (std 0.0176) using `SVC`.
- digits: `gemini-2.5-pro` — mean accuracy 0.9494 (std 0.00596) using `SVC`.
- iris: `gemini-2.5-pro` — mean accuracy 0.9200 (std 0.04) using `SVC`.

A few observations:
- The `pro` variant tends to produce slightly higher-scoring pipelines (wine, digits, iris) at the cost of longer generation times compared to `flash` and `flash_lite`.
- `Flash` and `flash_lite` sometimes produce well-performing pipelines with shorter generation time and longer code length (e.g., `flash_lite` often returns `GridSearchCV` with more detailed steps).

## Table 2 — Ablation Study (effects of pipeline components)
Key ablation effects aggregated by dataset (select highlights):

- Wine (n_runs=5, flash_lite pipeline):
  - `full` mean accuracy: 0.9722 ± 0.0393 (95% CI: [0.9234, 1.0210])
  - `no_scaling` mean accuracy: 0.7056 ± 0.0576
  - ANOVA: F=35.9460, p=2.1797e-10 (significant). Pairwise tests indicate `no_scaling` vs `full` is significant after Bonferroni correction (p=0.00154 < 0.003333).
  - Interpretation: Feature scaling is crucial for `wine` (SVC and some models are sensitive to unscaled features).

- Breast_cancer (n_runs=5, flash_lite pipeline):
  - `full` mean accuracy: 0.94035 ± 0.0190
  - `minimal` mean accuracy: 0.95965 ± 0.01819 (the minimal configuration surprisingly slightly outperformed full here)
  - `no_scaling` mean accuracy: 0.84737 ± 0.04625
  - ANOVA: F=13.7386, p=2.1859e-06 (significant). Pairwise comparisons: `full` vs `no_scaling` p=0.0042 (not Bonferroni-corrected) and `no_scaling` vs `no_feature_engineering` corrected p < 0.003333 significant.
  - Note: occasional `minimal > full` can occur when tuning/ensembling is not helpful on modest sample sizes and a simple model generalizes better.

- Digits (n_runs=5, flash_lite) and n_runs=10 aggregated:
  - `full` mean accuracy (flash_lite_n5): 0.9789 ± 0.0046
  - `no_scaling` mean accuracy (flash_lite_n5): 0.9022 ± 0.0177
  - ANOVA: F=95.5145, p=4.8820e-15 (significant). Pairwise comparisons show `full` vs `no_scaling` significant after correction (p < 0.003333). `no_feature_engineering` often performs near or slightly better than full (0.9822 > 0.9789 for digits in flash_lite pipelines), which suggests that for very high dimensional datasets (like digits with 64 features) PCA or simpler pipelines may not be necessary and tuned models alone provide strongest performance.

- Iris (n_runs=5 flash_lite, and n_runs=10 overall):
  - For `flash_lite` (n5) ANOVA was not significant (F=0.5417, p≈0.7428), indicating no clear difference across ablation settings at n=5.
  - For n=10 aggregated runs (not necessarily tied to a single variant), results showed `minimal` mean 0.9567 and `full` 0.9100 and an ANOVA p≈0.00286 indicating a statistically significant difference; in pairwise tests minimal vs full was significant in those runs.
  - Interpretation: For small datasets like `iris`, the effect of ablation is sensitive to sampling variance; ensemble and tuning may not always help depending on random seed and the structure of the dataset.

### New results at n=20 for Digits and Iris (updated)

- Digits (n_runs=20): the `minimal` and `no_feature_engineering` configurations now achieve mean accuracies ~0.9396 and ~0.9393 respectively, compared with `full` 0.9194. ANOVA: F=8.6138, p=5.987e-07; pairwise comparisons show `full` vs `minimal` and `no_feature_engineering` significant after Bonferroni correction. This strengthens the claim that feature engineering and additional tuning/ensembling may be unnecessary or detrimental for this dataset when using the generated pipelines.

- Iris (n_runs=20): `minimal` and `no_feature_engineering` both produce mean accuracies ~0.9567 compared with `full` 0.9117; ANOVA: F=8.5404, p=6.78e-07 and pairwise contrasts show `full` vs `minimal`/`no_feature_engineering` are significant (Bonferroni-corrected). This corroborates the n=10 result and indicates that a simple pipeline can outperform more complex pipelines on small datasets in this context.

## Statistical analysis & replication
- All statistical analyses (ANOVA and pairwise t-tests with Bonferroni correction) are executed in the `src/mle_star_ablation/stats.py` module and the outputs are stored next to each run in `results/<dataset>_<variant*` directories:
  - `summary_statistics_*.csv` — per-configuration summary metrics (mean, std, CI, n_runs)
  - `statistical_report_*.txt` — detailed ANOVA and pairwise results
  - `pairwise_comparisons_*.csv` — pairwise p-values and Cohen's d

Example to find ANOVA and pairwise p-values for `wine` (flash_lite_n5):
```pwsh
Get-Content results/wine_flash_lite_n5/flash_lite/statistical_report_20251117_180515.txt | select -first 60
```

## Figures & visualization
- We save per-experiment visualization (barplot, boxplot, violin, p-value heatmaps) under each run directory. For example: `results/wine_flash_lite_n5/flash_lite/` contains `comparison_barplot.png`, `comparison_boxplot.png`, `pvalue_heatmap.png`, and `cohend_heatmap.png`.

## Limitations and interpretation guidance
- Small datasets with a large number of pipeline choices can lead to results where simpler (minimal) pipelines outperform more complex (tuned/ensembled) ones due to overfitting during tuning.
- Some generated pipelines don't include specific steps (e.g., some pipelines use a preprocessor without a scaler) — in such cases, the corresponding ablation option is a no-op.
- Smoke runs (n=1) are reported separately in the aggregated CSV to show the generated pipelines but should not be used for statistical inference.

---

For the full CSVs and to reproduce these tables or export LaTeX-ready tables, see the `reports` folder and run the aggregation and cleaning commands in the Methodology section.
