# Full Cycle Report (20251113_212202)

## Configuration
Tasks: iris_gbdt
Cycles: 1
Artifact Types: initialization
Evaluation Repeat: 1
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
Latest Evaluation CSV: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\auto_evaluation_20251113_212200.csv`
Latest Evaluation JSON: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\auto_evaluation_20251113_212200.json`
Latest Visualization PNG: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\results_summary_20251113_232201.png`
Latest Visualization JSON: `D:\School\GoIT\MAUP_REDO_HWs\Diploma\model_comparison_results\results_summary_20251113_232201.json`

## Reproduction Command
```bash
python scripts/run_full_cycle.py --tasks iris_gbdt --cycles 1 --evaluation-repeat 1 --artifact-types initialization --visualize --report
```
