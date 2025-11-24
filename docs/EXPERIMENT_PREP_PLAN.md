# Experiment Preparation Plan

## Overview
- Goal: run the upcoming sweep on four regression datasets plus a California classification variant without surprises.
- Status: wrappers regenerated with estimator-aware tuning grids, smoke tests completed on `iris` (classification) and `california-housing-prices` (regression), pytest suite green.
- Blocking issues: none, but the placeholder regression datasets still require finalized CSV paths before the suite can execute.

## Wrapper & Environment Validation
- `python scripts/check_wrapper_importable.py --dir model_comparison_results` re-run on 2025-11-18: all `_pipeline_wrapper.py` files load successfully; raw Gemini scripts still fail as expected due to missing `generated_pipelines` or CSV inputs.
- ADK preflight already confirmed API credentials and Gemini model availability; no further action required before experiments.

## Experiment Suite Manifest
- New file `configs/experiment_suite.yaml` captures every experiment we plan to execute. Each entry tracks:
  - dataset name (`california-housing-prices` or `custom` for future regression CSVs),
  - pipeline wrapper path,
  - task type, seeds, and per-experiment overrides,
  - toggleable `enabled` flag for quick inclusion/exclusion.
- Four regression placeholders (`regression_custom_1` .. `regression_custom_4`) are disabled until CSV paths and target columns are ready.
- `california_classification_bins` reserved for any classification framing once binned labels exist.

## Batch Runner Utility
- Script: `scripts/run_experiment_suite.py` consumes the manifest and prints (dry run) or executes the corresponding `scripts/run_ablation.py` commands sequentially.
- Usage examples:
  - Dry run (default): `python scripts/run_experiment_suite.py` → prints all commands without running experiments.
  - Execute enabled experiments: `python scripts/run_experiment_suite.py --execute`.
  - Targeted subset: `python scripts/run_experiment_suite.py --only california_regression --execute`.
- Each experiment writes to `results/<experiment-name>/` to keep outputs separated and reproducible.

## Recommended Pre-Run Checklist
1. Fill `csv_path`/`target` for the four custom regression entries (or switch their `pipeline_file` values if different wrappers are preferred).
2. Decide whether the optional California classification variant is viable (requires a labeled/binned dataset) and enable it if so.
3. Review runtime budgets: current smoke tests (~3 runs/config) finish in under 30 seconds per dataset; the full sweep (5+ runs, 6 configs) scales linearly.
4. Once everything is ready, run `python scripts/run_experiment_suite.py` to verify commands, then re-run with `--execute` when approved.

## Notes
- `california-housing-prices` "no_scaling" ablation collapses to near-zero R² because only a single raw feature survives; this is expected for the baseline config.
- Keep the `--no-plots` flag for headless environments; remove it if you need PDF/PNG outputs for reports.
- If new wrappers are generated later, re-run `scripts/check_wrapper_importable.py` before executing the suite.
