# Ablation Experiments Summary

**Date:** 2025-11-17

This document summarizes the ablation analysis results completed in the 4-day plan (wine, iris, digits; breast_cancer done earlier).

## Overview
- Datasets analyzed: `breast_cancer`, `wine`, `iris`, `digits`
- Configurations: `full`, `no_scaling`, `no_feature_engineering`, `no_tuning`, `no_ensemble`, `minimal`
- Repeats: 5 runs per configuration
- CV: Stratified 5-fold

## Key Findings

### Wine (`results/wine_n5`):
- Baseline (`full`) mean accuracy: 0.9556
- `no_scaling` mean accuracy: 0.7222
- Δ(no_scaling) = -0.2333 (−23.33 percentage points) vs `full`
- ANOVA p-value: 4.96e-11 (statistically significant)
- Pairwise comparisons show `no_scaling` is significantly worse than all other configurations (Bonferroni corrected)
- Conclusion: **Scaling is CRITICAL for wine**. The claim in the demo report that scaling impacts wine is verified.

### Iris (`results/iris_n5`):
- Baseline (`full`) mean accuracy: 0.9200
- `minimal` mean accuracy: 0.9400
- Δ(minimal) = +0.0200 (2 percentage points) vs `full`
- ANOVA p-value: 0.567 (not statistically significant)
- Pairwise comparisons not significant after Bonferroni correction
- Conclusion: **Minimal pipeline appears to have higher mean accuracy for Iris**, but evidence is **not statistically significant** at corrected alpha; the demo claim "minimal surpasses full" is **not robust** and must be stated with caveats.

### Digits (`results/digits_n5`):
- Baseline (`full`) mean accuracy: 0.9222
- `minimal` mean accuracy: 0.9350
- Δ(minimal) = +0.0128 (1.28 percentage points) vs `full`
- ANOVA p-value: 0.355 (not significant)
- Conclusion: Data do not indicate statistically significant differences among configurations for digits.

### Breast Cancer (completed earlier):
- Baseline (`full`) mean accuracy: 0.9702
- `no_scaling` mean accuracy: 0.9140
- Δ(no_scaling) = -0.0562 (−5.62 percentage points)
- ANOVA p-value: < 0.001, significant
- Conclusion: Scaling is critical for breast_cancer (already verified earlier)

## Implications for the Demo Report Statements
- The claim that **scaling is critical** is **verified** for `breast_cancer` and `wine`.
- The claim that **minimal pipeline outperforms full** is **not robust**: observed on `iris` in mean, but not statistically significant.
- The claim that **minimal pipelines can retain >95% accuracy** is **true** for some datasets (e.g., `breast_cancer`, `iris`, `digits`), but should be qualified: depends on dataset and randomness.

## Next Steps
1. Add the results and plots to the final report (`results/...` and `reports/ablation_summary.md`).
2. If you want to tighten significance tests, increase `n_runs` from 5 to 10 for critical comparisons (breast_cancer already significant; wine borderline but clear; iris/digits might need more repeats).
3. Document the exact commands used and link to `results/` directories for reproducibility.

## Commands Used

```powershell
# Wine ablation (5 repeats)
& "C:/Program Files/Python313/python.exe" scripts/run_ablation.py --dataset wine --n-runs 5 --output-dir results/wine_n5 --verbose > logs/wine_ablation.log 2>&1 &

# Iris ablation (5 repeats)
& "C:/Program Files/Python313/python.exe" scripts/run_ablation.py --dataset iris --n-runs 5 --output-dir results/iris_n5 --verbose > logs/iris_ablation.log 2>&1 &

# Digits ablation (5 repeats)
& "C:/Program Files/Python313/python.exe" scripts/run_ablation.py --dataset digits --n-runs 5 --output-dir results/digits_n5 --verbose > logs/digits_ablation.log 2>&1 &
```

## Notes
- All outputs are saved in `results/<dataset>_n5/`, including CSVs and plots.
- We used the default 6 standard configs for ablation; a YAML config could be added to select specific configs if needed.

---

*Prepared by: GitHub Copilot*