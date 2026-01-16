# Full Cycle Report (20251113_191617)

## Configuration
Tasks: iris_gbdt
Cycles: 1
Artifact Types: initialization
Evaluation Repeat: 1
Skip Run: False
Skip Eval: False

## Summary Metrics
- california-housing-prices/initialization: 206171.2917

## Pipeline Status
| Task | Initialization | Refinement | Ensemble | Status |
|------|----------------|------------|----------|--------|
| california-housing-prices | ✅ | ❌ | ❌ | Initialization only |

## Artifacts
Manifest: `generated_pipelines/mle_star_extraction_manifest.json`
Latest Evaluation CSV: `model_comparison_results/auto_evaluation_20251113_191615.csv`
Latest Evaluation JSON: `model_comparison_results/auto_evaluation_20251113_191615.json`
Latest Visualization PNG: `model_comparison_results/results_summary_20251113_211616.png`
Latest Visualization JSON: `model_comparison_results/results_summary_20251113_211616.json`

## Reproduction Command
```bash
python scripts/run_full_cycle.py --tasks iris_gbdt --cycles 1 --evaluation-repeat 1 --artifact-types initialization --visualize --report
```
