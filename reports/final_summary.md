# Final Summary of Ablation Experiments and Model Comparisons

**Date:** 2025-11-17

This file summarizes key findings from ablation experiments and model comparisons performed on the `breast_cancer`, `wine`, `digits`, and `iris` datasets across Gemini variants (Flash Lite, Flash, Pro).

## Key Findings

- **Scaling (StandardScaler)** is critical for some datasets (e.g., `breast_cancer` and `wine`), as removing scaling drops accuracy significantly (e.g., `wine`: 95.56 → 72.22 mean accuracy, ANOVA p < 0.001).
- **Minimal pipelines** (`minimal` configuration) can achieve near-baseline accuracy for some datasets (e.g., `breast_cancer`, `iris`, `digits`), and in some cases show higher mean values than `full` (e.g., `iris` and `digits`), though statistical significance depends on number of runs and corrections.
- For **iris**, after increased repetitions (n_runs=10), overall ANOVA is significant (p≈0.00286) showing measurable differences between configurations, but pairwise corrected significance is more conservative due to Bonferroni.
- For **digits**, after n_runs=10, ANOVA indicates significant differences (p≈0.0218) and pairwise tests show `minimal` and `no_feature_engineering` are significantly better than `full` with corrected tests for some pairs.
- **Gemini variants**: `Flash Lite` provides a good tradeoff (fast generation + competitive accuracy), while `Pro` sometimes chooses different models/pipeline structures. Detailed variant comparisons are available in `reports/table1_model_comparison.csv`.

## Recommendations for the Report

1. **Include both model comparison (Table 1) and ablation analysis (Table 2)** in the results section. Use `reports/table1_model_comparison.csv` and `reports/table2_ablation.csv` as tables in the report.
2. **Be precise about statistical claims**: if claiming a configuration is better/worse, present ANOVA p-values and corrected pairwise p-values from the `statistical_report_*.txt` files.
3. **Report variant differences** (Flash Lite vs Pro) and justify the trade-offs (generation time vs accuracy).
4. **Declare limitations and honesty**: When a claim is not strongly supported by data (e.g., minimal > full), say it is a mean effect requiring more repetitions or different seed control.

## Next Steps (for publication-level analysis)

- Run `n_runs=20` for datasets where results are near-significant (iris, digits)
- Perform component-wise ablation (imputer-only vs scaler-only vs different scalers)
- Reproduce results with other LLMs (GPT-4, Claude) if needed for publication extension

## Files Generated

- `reports/table1_model_comparison.csv` — model/variant accuracy and meta
- `reports/table2_ablation.csv` — ablation mean/std/ci for dataset/variant/config
- `reports/aggregate_summary_stats.csv` — raw aggregation used to build Table 2
- `results/*` — per-run detailed outputs and statistical reports

---

Prepared by GitHub Copilot
