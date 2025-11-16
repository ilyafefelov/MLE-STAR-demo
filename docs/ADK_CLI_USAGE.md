## ADK CLI Interactive Usage

This guide explains how to interact with the MLE-STAR ADK CLI for the `machine_learning_engineering` agent when you run:

```bash
adk run machine_learning_engineering
```

### Execution Model
The ADK spawns a multi-agent sequence:
1. Initialization Agent
2. Refinement Agent (iterative loops)
3. Ensemble Agent
4. Robustness / Validation phase

Each agent may ask for approval or additional instructions depending on configuration. In the default sample repo configuration, most steps run autonomously unless the system encounters an error.

### Prompts & Input
During execution you may see a line prefix like:
```
[user]:
```
This indicates the framework is ready to accept optional human guidance. If you do not provide input and just press Enter, the agent continues with its internal plan.

You can inject guidance such as:
```
Focus on gradient-boosted trees and avoid deep models.
```

### Termination Conditions
The run is considered complete when the final agent prints its summary and the process exits with code 0. If you only see initialization artifacts (e.g. `init_code_1.py`) in the workspace and the process terminated early, likely a quota or API error halted progress.

Indicators of early termination:
- No `train*.py` or `refined*.py` files appear in the run folder.
- Workspace lacks a final summary or submission artifacts.
- Logs contain API error codes (e.g., HTTP 429).

### Restarting After Failure
1. Kill the hung or failed process if still running.
2. Inspect logs under `adk-samples/python/agents/machine-learning-engineering/logs/`.
3. Re-run `adk run machine_learning_engineering` after addressing the cause (e.g., increase API quota or switch model variant).

### Switching Tasks
`shared_libraries/config.py` contains `task_name`. Modify it before launching ADK for a different dataset/task variant (e.g., `iris_gbdt`). Batch automation scripts pass task names directly and bypass manual editing.

### Non-Interactive Batch Mode
Using `scripts/run_mle_star_batch.py` executes tasks without requiring interactive input. Combine with the orchestrator `run_full_cycle.py` to achieve full automation.

### Common Issues
| Symptom | Cause | Resolution |
|---------|-------|-----------|
| 429 RESOURCE_EXHAUSTED | API quota limit | Reduce parallelism / upgrade quota / switch model tier |
| Missing refinement files | Early failure post-init | Check logs, re-run cycle |
| Relative path errors (`./input/train.csv` not found) | Running artifact outside workspace | Always evaluate in original source directory (the evaluation script already enforces this) |

### Verifying Success
Successful full runs usually produce multiple Python files beyond initialization (train/refined/ensemble) under `workspace/<task>/<run_id>/` and potentially model candidate descriptions. Use the extraction + evaluation scripts afterward for structured experiment logging.

---
This usage reference complements `AUTOMATED_EXPERIMENTS.md` and clarifies interactive vs automated operation modes.
