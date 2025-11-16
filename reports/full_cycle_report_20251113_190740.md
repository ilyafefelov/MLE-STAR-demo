# Full Cycle Report (20251113_190740)

## Configuration
Tasks: iris_gbdt
Cycles: 1
Artifact Types: initialization, refinement, ensemble, pipeline
Evaluation Repeat: 2
Skip Run: False
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
Manifest: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\generated_pipelines\mle_star_extraction_manifest.json`
Latest Evaluation CSV: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\auto_evaluation_20251113_190739.csv`
Latest Evaluation JSON: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\auto_evaluation_20251113_190739.json`
Latest Visualization PNG: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\results_summary_20251113_210740.png`
Latest Visualization JSON: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\results_summary_20251113_210740.json`

## Reproduction Command
```bash
python scripts/run_full_cycle.py --tasks iris_gbdt --cycles 1 --evaluation-repeat 2 --artifact-types initialization refinement ensemble pipeline --visualize --report
```
