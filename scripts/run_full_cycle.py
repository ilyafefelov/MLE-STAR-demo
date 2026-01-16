#!/usr/bin/env python
"""End-to-end automated experiment orchestrator.

Sequence per cycle:
    1. Run MLE-STAR tasks via `run_mle_star_batch.py` (unless --skip-run)
    2. Extract artifacts via `extract_mle_star_pipelines.py`
    3. Evaluate artifacts via `evaluate_extracted_artifacts.py` (unless --skip-eval)
    4. (Optional) Visualize latest evaluation with `visualize_results.py` (--visualize)
    5. (Optional) Generate Markdown report summarizing run (--report)

Supports multiple cycles to study stability / reproducibility.
Each cycle appends new entries to extraction manifest and produces new evaluation CSV/JSON.

No fallback code is ever injected; only ADK outputs are used.

Usage examples:
    python scripts/run_full_cycle.py --gbdt --cycles 2 --evaluation-repeat 3 --visualize --report
    python scripts/run_full_cycle.py --tasks iris_gbdt --cycles 1 --artifact-types initialization refinement ensemble pipeline --evaluation-repeat 2 --visualize
    python scripts/run_full_cycle.py --skip-run --cycles 1 --evaluation-repeat 2 --visualize --report  # re-evaluate existing artifacts

"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
import time
import json
import re
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
GENERATED_DIR = REPO_ROOT / "generated_pipelines"
MANIFEST_NAME = "mle_star_extraction_manifest.json"
MANIFEST_PATH = GENERATED_DIR / MANIFEST_NAME


def run_step(cmd: list[str], cwd: Path | None = None, label: str = "STEP") -> int:
    print(f"\n=== {label} ===")
    print("Command:", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    dur = time.time() - start
    print(f"=== {label} finished (exit={proc.returncode}, {dur:.1f}s) ===")
    return proc.returncode


def get_latest_log_tail(log_dir: Path, task_name: str, tail_lines: int = 30) -> str:
    """Retrieve tail of latest log file for a task."""
    if not log_dir.exists():
        return "(Log directory not found)"
    logs = sorted(log_dir.glob(f"{task_name}_*.log"))
    if not logs:
        return "(No log file found)"
    latest_log = logs[-1]
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        tail = ''.join(lines[-tail_lines:])
        return f"Last {tail_lines} lines from {latest_log.name}:\n{tail}"
    except Exception as e:
        return f"(Error reading log: {e})"


def parse_latest_evaluation(output_dir: Path) -> dict:
    if not output_dir.exists():
        return {}
    csv_files = sorted(output_dir.glob("auto_evaluation_*.csv"))
    if not csv_files:
        return {}
    latest = csv_files[-1]
    summary = {}
    # Simple aggregation: mean per task/artifact_type
    pattern = re.compile(r",")
    with open(latest, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_task = header.index("task") if "task" in header else None
        idx_type = header.index("artifact_type") if "artifact_type" in header else None
        idx_val = header.index("metric_value") if "metric_value" in header else None
        if None in (idx_task, idx_type, idx_val):
            return {}
        for line in f:
            parts = pattern.split(line.strip())
            if len(parts) < max(idx_task, idx_type, idx_val) + 1:
                continue
            task = parts[idx_task]
            a_type = parts[idx_type]
            try:
                val = float(parts[idx_val])
            except ValueError:
                continue
            key = f"{task}/{a_type}"
            summary.setdefault(key, []).append(val)
    return {k: sum(v) / len(v) for k, v in summary.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run full automatic experiment cycles (run→extract→evaluate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks", nargs="*", default=None, help="Explicit task list (overrides --gbdt)")
    p.add_argument("--gbdt", action="store_true", help="Shortcut for iris_gbdt wine_gbdt digits_gbdt")
    p.add_argument("--cycles", type=int, default=1, help="Number of full cycles to execute")
    p.add_argument("--parallel", action="store_true", help="Run tasks in parallel in batch step")
    p.add_argument("--evaluation-repeat", type=int, default=1, help="Repeat each artifact execution this many times")
    p.add_argument("--artifact-types", nargs="*", default=None, help="Filter artifact types for evaluation")
    p.add_argument("--timeout-eval", type=int, default=180, help="Per-script timeout for evaluation")
    p.add_argument("--skip-run", action="store_true", help="Skip running ADK (use existing workspace)")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation (only run + extract)")
    p.add_argument("--include-candidates", action="store_true", help="Include model candidate descriptions in extraction")
    p.add_argument("--fail-fast", action="store_true", help="Abort cycle on first non-zero exit code")
    p.add_argument("--visualize", action="store_true", help="Generate visualization PNG/JSON after evaluation")
    p.add_argument("--report", action="store_true", help="Generate Markdown report summarizing latest evaluation")
    p.add_argument("--report-dir", default="reports", help="Directory for generated Markdown reports")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.tasks:
        task_list = args.tasks
    elif args.gbdt:
        task_list = ["iris_gbdt", "wine_gbdt", "digits_gbdt"]
    else:
        task_list = ["breast_cancer", "wine", "digits", "iris"]

    print("=" * 80)
    print("FULL AUTOMATED EXPERIMENT CYCLE")
    print("=" * 80)
    print(f"Tasks: {', '.join(task_list)}")
    print(f"Cycles: {args.cycles}")
    print(f"Parallel batch: {args.parallel}")
    print(f"Eval repeats: {args.evaluation_repeat}")
    if args.artifact_types:
        print(f"Artifact types filter: {', '.join(args.artifact_types)}")
    print(f"Skip run: {args.skip_run}  Skip eval: {args.skip_eval}")
    print(f"Visualize: {args.visualize}  Report: {args.report}")
    print("=" * 80)

    SCRIPTS_DIR.mkdir(exist_ok=True, parents=True)
    GENERATED_DIR.mkdir(exist_ok=True, parents=True)

    run_failures = []  # Track failed tasks
    adk_log_dir = REPO_ROOT / "adk-samples" / "python" / "agents" / "machine-learning-engineering" / "logs"

    for cycle in range(1, args.cycles + 1):
        print(f"\n===== CYCLE {cycle}/{args.cycles} =====")
        # 1. Run ADK batch
        if not args.skip_run:
            batch_cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "run_mle_star_batch.py"),
                "--tasks",
                *task_list,
            ]
            if args.parallel:
                batch_cmd.append("--parallel")
            exit_code = run_step(batch_cmd, label="RUN-MLE-STAR")
            if exit_code != 0:
                for task in task_list:
                    log_tail = get_latest_log_tail(adk_log_dir, task)
                    run_failures.append({
                        "task": task,
                        "exit_code": exit_code,
                        "log_tail": log_tail
                    })
                if args.fail_fast:
                    print("Aborting due to failure in RUN-MLE-STAR step.")
                    return
        else:
            print("(Skipping run step)")

        # 2. Extract artifacts
        extract_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "extract_mle_star_pipelines.py"),
        ]
        if args.include_candidates:
            extract_cmd.append("--include-candidates")
        exit_code = run_step(extract_cmd, label="EXTRACT-ARTIFACTS")
        if exit_code != 0 and args.fail_fast:
            print("Aborting due to failure in EXTRACT-ARTIFACTS step.")
            return

        # 3. Evaluate artifacts
        if not args.skip_eval:
            eval_cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_extracted_artifacts.py"),
                "--manifest",
                str(MANIFEST_PATH),
                "--repeat",
                str(args.evaluation_repeat),
                "--timeout",
                str(args.timeout_eval),
            ]
            if args.artifact_types:
                eval_cmd.extend(["--artifact-types", *args.artifact_types])
            exit_code = run_step(eval_cmd, label="EVALUATE-ARTIFACTS")
            if exit_code != 0 and args.fail_fast:
                print("Aborting due to failure in EVALUATE-ARTIFACTS step.")
                return
            summary = parse_latest_evaluation(REPO_ROOT / "model_comparison_results")
            if summary:
                print("\nCycle summary (mean metrics):")
                for k, v in sorted(summary.items()):
                    print(f"  {k}: {v:.4f}")
            else:
                print("\nNo evaluation summary available for this cycle.")

            # 4. Visualization (latest evaluation)
            if args.visualize:
                viz_cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "visualize_results.py"),
                ]
                exit_code = run_step(viz_cmd, label="VISUALIZE-RESULTS")
                if exit_code != 0 and args.fail_fast:
                    print("Aborting due to failure in VISUALIZE-RESULTS step.")
                    return

            # 5. Markdown report
            if args.report:
                report_dir = REPO_ROOT / args.report_dir
                report_dir.mkdir(exist_ok=True, parents=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                report_path = report_dir / f"full_cycle_report_{ts}.md"
                latest_eval_csv = sorted((REPO_ROOT / "model_comparison_results").glob("auto_evaluation_*.csv"))
                latest_eval_json = sorted((REPO_ROOT / "model_comparison_results").glob("auto_evaluation_*.json"))
                latest_viz_png = sorted((REPO_ROOT / "model_comparison_results").glob("results_summary_*.png"))
                latest_viz_json = sorted((REPO_ROOT / "model_comparison_results").glob("results_summary_*.json"))
                
                # Determine pipeline status per task from manifest
                pipeline_status = {}
                if MANIFEST_PATH.exists():
                    try:
                        manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                        for entry in manifest_data.get("entries", []):
                            task = entry.get("task")
                            artifact_type = entry.get("artifact_type")
                            if task not in pipeline_status:
                                pipeline_status[task] = {"initialization": False, "refinement": False, "ensemble": False}
                            if artifact_type in pipeline_status[task]:
                                pipeline_status[task][artifact_type] = True
                    except Exception:
                        pass
                
                lines = [
                    f"# Full Cycle Report ({ts})",
                    "",
                    "## Configuration",
                    f"Tasks: {', '.join(task_list)}",
                    f"Cycles: {args.cycles}",
                    f"Artifact Types: {', '.join(args.artifact_types) if args.artifact_types else 'ALL'}",
                    f"Evaluation Repeat: {args.evaluation_repeat}",
                    f"Skip Run: {args.skip_run}",
                    f"Skip Eval: {args.skip_eval}",
                    "",
                    "## Summary Metrics",
                ]
                if summary:
                    for k, v in sorted(summary.items()):
                        lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append("(No metrics available)")
                lines.append("")
                
                # Pipeline status table
                lines.append("## Pipeline Status")
                if pipeline_status:
                    lines.append("| Task | Initialization | Refinement | Ensemble | Status |")
                    lines.append("|------|----------------|------------|----------|--------|")
                    for task, stages in sorted(pipeline_status.items()):
                        init = "✅" if stages["initialization"] else "❌"
                        refine = "✅" if stages["refinement"] else "❌"
                        ensemble = "✅" if stages["ensemble"] else "❌"
                        if stages["initialization"] and stages["refinement"] and stages["ensemble"]:
                            status = "Complete"
                        elif stages["initialization"] and stages["refinement"]:
                            status = "Partial (no ensemble)"
                        elif stages["initialization"]:
                            status = "Initialization only"
                        else:
                            status = "No artifacts"
                        lines.append(f"| {task} | {init} | {refine} | {ensemble} | {status} |")
                else:
                    lines.append("(No pipeline status available)")
                lines.append("")
                
                # Failure diagnostics
                if run_failures:
                    lines.append("## Run Failures")
                    for failure in run_failures:
                        lines.append(f"### Task: {failure['task']} (exit code: {failure['exit_code']})")
                        lines.append("```")
                        lines.append(failure['log_tail'])
                        lines.append("```")
                        lines.append("")
                
                lines.append("## Artifacts")
                if MANIFEST_PATH.exists():
                    lines.append(f"Manifest: `{MANIFEST_PATH}`")
                if latest_eval_csv:
                    lines.append(f"Latest Evaluation CSV: `{latest_eval_csv[-1]}`")
                if latest_eval_json:
                    lines.append(f"Latest Evaluation JSON: `{latest_eval_json[-1]}`")
                if latest_viz_png:
                    lines.append(f"Latest Visualization PNG: `{latest_viz_png[-1]}`")
                if latest_viz_json:
                    lines.append(f"Latest Visualization JSON: `{latest_viz_json[-1]}`")
                lines.append("")
                lines.append("## Reproduction Command")
                repro_cmd = [
                    "python", "scripts/run_full_cycle.py",
                    "--tasks", *task_list,
                    "--cycles", str(args.cycles),
                    "--evaluation-repeat", str(args.evaluation_repeat),
                ]
                if args.artifact_types:
                    repro_cmd.extend(["--artifact-types", *args.artifact_types])
                if args.visualize:
                    repro_cmd.append("--visualize")
                if args.report:
                    repro_cmd.append("--report")
                if args.skip_run:
                    repro_cmd.append("--skip-run")
                if args.skip_eval:
                    repro_cmd.append("--skip-eval")
                lines.append("```bash")
                lines.append(" ".join(repro_cmd))
                lines.append("```")
                lines.append("")
                report_path.write_text("\n".join(lines), encoding="utf-8")
                print(f"Report written: {report_path}")
        else:
            print("(Skipping evaluation step)")

    print("\n=== All cycles complete ===")
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            print(f"Manifest entries total: {manifest.get('total_entries')} (file: {MANIFEST_PATH})")
        except Exception:
            print("Manifest read error; skipping.")
    print("Done.")


if __name__ == "__main__":
    main()
