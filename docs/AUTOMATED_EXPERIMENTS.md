## Automated Experiment Workflow

This document describes the fully automated workflow for running MLE-STAR tasks, extracting generated artifacts, and evaluating them repeatedly without manual fallback code.

### Goals
- Zero manual edits between cycles.
- Reproducible multi-run experiment cycles for selected tasks.
- Consolidated metrics (CSV + JSON) for each artifact type (initialization/train/refinement/ensemble/pipeline).

### Components
- `scripts/run_mle_star_batch.py`: Launches ADK MLE-STAR agents per task (sequential or parallel).
- `scripts/extract_mle_star_pipelines.py`: Harvests Python code + candidate descriptions from workspace runs into `generated_pipelines/` and produces a manifest.
- `scripts/evaluate_extracted_artifacts.py`: Executes each artifact in its original source directory, parses printed metrics, and writes aggregated results.

### Minimal End-to-End Cycle
```bash
# 1. Run selected tasks (example: GBDT tasks sequential)
python scripts/run_mle_star_batch.py --gbdt

# 2. Extract all artifacts (creates manifest with metadata)
python scripts/extract_mle_star_pipelines.py --include-candidates

# 3. Evaluate all extracted Python artifacts (repeat each 3 times)
python scripts/evaluate_extracted_artifacts.py \
  --manifest generated_pipelines/mle_star_extraction_manifest.json \
  --repeat 3 --timeout 180
```

### Filtering Examples
```bash
# Only evaluate initialization + pipeline artifacts for iris_gbdt run 2
python scripts/evaluate_extracted_artifacts.py \
  --manifest generated_pipelines/mle_star_extraction_manifest.json \
  --tasks iris_gbdt --runs 2 \
  --artifact-types initialization pipeline
```

### Output Artifacts
| Location | Description |
|----------|-------------|
| `generated_pipelines/` | Copied code files, renamed `<task>_run<id>_<original>.py` |
| `generated_pipelines/mle_star_extraction_manifest.json` | Manifest with provenance metadata |
| `model_comparison_results/auto_evaluation_<timestamp>.csv` | Parsed metric rows |
| `model_comparison_results/auto_evaluation_<timestamp>.json` | Detailed per-execution results |

### Metric Parsing Heuristics
The evaluator searches stdout lines for case-insensitive keywords:
`Accuracy`, `Final Validation Performance`, `score`.
It extracts the first floating number on those lines. If multiple lines match, each becomes a separate metric record.

### Repetition & Stability
Use `--repeat N` to re-run each artifact script N times. This helps surface variance due to stochastic estimators or random splits present in generated code. Each repetition is indexed and stored separately.

### Error Handling
- Non-zero script exit codes are logged; metrics may still be parsed if printed before failure.
- Timeouts (`--timeout`) prevent hanging on long training loops.
- Use `--fail-fast` to abort the entire evaluation on first error.

### Performance Considerations
- Parallelization of evaluation is currently not implemented to keep logs deterministic. A future enhancement could introduce a `--jobs` flag using multiprocessing.
- Long-running refinement loops may exceed the default timeout; raise `--timeout` selectively if needed.

### Extending the Workflow
Potential next steps (not yet implemented):
- Orchestrator script combining run → extract → evaluate in one command.
- Automatic comparison across cycles (e.g., stability tracking per task/artifact type).
- Additional metric parsing (F1, ROC AUC) if generated scripts print them.

### Troubleshooting
| Issue | Cause | Mitigation |
|-------|-------|------------|
| Missing metrics | Generated code prints no recognizable lines | Add prompt engineering upstream or extend regex patterns |
| Relative CSV path errors | Artifact expects `./input/train.csv` but directory differs | Evaluator runs in original source directory to preserve paths |
| TimeoutExpired | Long training or network wait | Increase `--timeout` |
| Empty manifest | Extraction not run or no artifacts | Re-run extraction after ADK batch run |

### Clean Slate Re-Run
To start a fresh cycle, you may archive or remove previous workspace run folders before launching new agents to avoid ambiguity in extraction.

```bash
# Optional: clean prior generated pipelines (archive first if needed)
mkdir -p archive/$(date +%Y%m%d_%H%M%S)
mv generated_pipelines/*.py archive/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
```

### Versioning & Provenance
Each manifest entry includes the original absolute source path. This allows tracing back to the exact run folder under `workspace/<task>/<run_id>/` for audit or code regeneration.

---
This workflow ensures fully automatic experiment execution without relying on manually created fallback baselines. All evaluated code originates from genuine ADK agent outputs.
