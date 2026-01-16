# Full Cycle Report (20251113_173835)

## Configuration
Tasks: iris_gbdt
Cycles: 1
Artifact Types: initialization, refinement, ensemble, pipeline
Evaluation Repeat: 2
Skip Run: True
Skip Eval: False

## Summary Metrics
- california-housing-prices/initialization: 206171.2917
- iris_gbdt/initialization: 0.9583

## Pipeline Status
| Task | Initialization | Refinement | Ensemble | Status |
|------|----------------|------------|----------|--------|
| california-housing-prices | ✅ | ❌ | ❌ | Initialization only |
| iris_gbdt | ✅ | ❌ | ❌ | Initialization only |

## Artifacts
Manifest: `generated_pipelines/mle_star_extraction_manifest.json`
Latest Evaluation CSV: `model_comparison_results/auto_evaluation_20251113_173833.csv`
Latest Evaluation JSON: `model_comparison_results/auto_evaluation_20251113_173833.json`
Latest Visualization PNG: `model_comparison_results/results_summary_20251113_193834.png`
Latest Visualization JSON: `model_comparison_results/results_summary_20251113_193834.json`

## Reproduction Command
```bash
python scripts/run_full_cycle.py --tasks iris_gbdt --cycles 1 --evaluation-repeat 2 --artifact-types initialization refinement ensemble pipeline --visualize --report --skip-run
```
