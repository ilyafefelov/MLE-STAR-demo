# Ablation Experiments Progress Log

This file tracks the progress of ablation experiments across datasets.

| Dataset | Started At | Finished At | Status | Runs (n_runs) | Notes |
|---------|------------|-------------|--------|---------------|-------|
| breast_cancer | 2025-11-13 12:50 | 2025-11-13 13:10 | Completed | 5 | Baseline ablation completed; results in `results/` |
| wine | 2025-11-17 16:23 | 2025-11-17 16:35 | Completed | 5 | Full ablation completed; results in `results/wine_n5/`; `no_scaling` significantly worse (Δ=-23.33 p<0.001) |
| iris | 2025-11-17 16:30 | 2025-11-17 16:36 | Completed | 5 | Full ablation completed; minimal > full in mean accuracy (0.94 vs 0.92) but ANOVA not significant (p=0.567); rerun with n_runs=10 started 2025-11-17 17:10 (results/iris_n10) |
| digits | 2025-11-17 16:35 | 2025-11-17 16:38 | Completed | 5 | Full ablation completed; differences not significant (ANOVA p=0.355); rerun with n_runs=10 started 2025-11-17 17:12 (results/digits_n10) |


## Log updates
- 2025-11-17: Created progress log and started smoke test for wine.
