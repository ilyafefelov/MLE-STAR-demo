#!/usr/bin/env python
"""Automated evaluation of extracted MLE-STAR artifacts.

Reads the manifest produced by `extract_mle_star_pipelines.py`, executes each
Python artifact in its original source directory (to preserve relative paths),
parses printed metrics (accuracy style), and aggregates results into a CSV + JSON.

Metric extraction heuristics:
  - Lines containing 'Accuracy' or 'Final Validation Performance' or 'score'
  - First floating number in that line captured as metric value
  - Multiple matches per file stored separately

Output:
  model_comparison_results/auto_evaluation_<timestamp>.csv
  model_comparison_results/auto_evaluation_<timestamp>.json

Usage:
  python scripts/evaluate_extracted_artifacts.py \
      --manifest generated_pipelines/mle_star_extraction_manifest.json \
      --repeat 3 --timeout 120

Optional flags:
  --artifact-types initialization train refinement ensemble pipeline
  --tasks iris_gbdt wine_gbdt
  --runs 2 3

Author: Фефелов Ілля Олександрович
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

METRIC_PATTERNS = [
    re.compile(r"accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"final validation performance\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"mse\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"rmse\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"mean squared error\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
]

FLOAT_FALLBACK = re.compile(r"([0-9]*\.?[0-9]+)")

DEFAULT_ARTIFACT_TYPES = {"initialization", "train", "refinement", "ensemble", "pipeline"}


@dataclass
class ArtifactResult:
    task: str
    run_id: str
    artifact_type: str
    source: str
    metric_name: str
    metric_value: float
    repeat_index: int
    raw_line: str
    stdout_snippet: str
    task_type: str = "unknown"


def load_task_metadata(config_path: Path) -> Dict:
    """Load task metadata from YAML config."""
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('tasks', {})


def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate extracted MLE-STAR artifacts automatically",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="Path to extraction manifest JSON")
    p.add_argument("--tasks", nargs="*", default=None, help="Filter by task names")
    p.add_argument("--runs", nargs="*", default=None, help="Filter by run ids")
    p.add_argument("--artifact-types", nargs="*", default=None, help="Filter by artifact types")
    p.add_argument("--repeat", type=int, default=1, help="Number of times to execute each artifact")
    p.add_argument("--timeout", type=int, default=180, help="Per-process timeout (seconds)")
    p.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    p.add_argument("--output-dir", default="model_comparison_results", help="Destination directory for results")
    p.add_argument("--fail-fast", action="store_true", help="Abort on first execution error")
    return p


def read_manifest(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics_from_stdout(stdout: str, task_type: str = "unknown") -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    lines = stdout.splitlines()
    for line in lines:
        lowered = line.lower()
        # Check for any metric keyword
        if any(key in lowered for key in ["accuracy", "final validation performance", "score", "mse", "rmse", "mean squared error"]):
            # Try regex patterns first
            found = False
            for pattern in METRIC_PATTERNS:
                m = pattern.search(lowered)
                if m:
                    try:
                        val = float(m.group(1))
                        name = pattern.pattern.split("\\s*")[0].replace("\\s*", "")
                        metrics.append({"name": name, "value": val, "line": line})
                        found = True
                        break
                    except ValueError:
                        pass
            if not found:
                # fallback: first float in line
                m2 = FLOAT_FALLBACK.search(line)
                if m2:
                    try:
                        val = float(m2.group(1))
                        metrics.append({"name": "generic", "value": val, "line": line})
                    except ValueError:
                        pass
    return metrics


def run_artifact(source: Path, python_exec: str, timeout: int) -> subprocess.CompletedProcess:
    # Run inside the artifact's directory to preserve relative CSV paths
    return subprocess.run(
        [python_exec, str(source)],
        cwd=source.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def evaluate_entry(entry: Dict, args, results: List[ArtifactResult], task_metadata: Dict) -> None:
    source_path = Path(entry["source"])
    if not source_path.exists():
        print(f"⚠️  Source missing (skipping): {source_path}")
        return
    task_name = entry["task"]
    task_info = task_metadata.get(task_name, {})
    task_type = task_info.get('type', 'unknown')
    for r in range(args.repeat):
        try:
            cp = run_artifact(source_path, args.python, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout: {source_path}")
            if args.fail_fast:
                raise
            continue
        stdout = cp.stdout
        if cp.returncode != 0:
            print(f"❌ Error ({cp.returncode}) running {source_path.name}")
            if args.fail_fast:
                print(stdout)
                raise RuntimeError(f"Failure in {source_path}")
        metrics = extract_metrics_from_stdout(stdout, task_type)
        if not metrics:
            print(f"🔍 No metrics found in {source_path.name}")
        snippet = "\n".join(stdout.splitlines()[-8:])  # tail snippet
        for m in metrics:
            results.append(
                ArtifactResult(
                    task=entry["task"],
                    run_id=entry["run_id"],
                    artifact_type=entry["artifact_type"],
                    source=entry["source"],
                    metric_name=m["name"],
                    metric_value=m["value"],
                    repeat_index=r,
                    raw_line=m["line"],
                    stdout_snippet=snippet,
                    task_type=task_type,
                )
            )


def write_outputs(results: List[ArtifactResult], outdir: Path) -> Path:
    outdir.mkdir(exist_ok=True, parents=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"auto_evaluation_{ts}.csv"
    json_path = outdir / f"auto_evaluation_{ts}.json"
    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "run_id", "artifact_type", "source", "metric_name", "metric_value", "repeat_index", "raw_line", "task_type"
        ])
        for r in results:
            w.writerow([
                r.task, r.run_id, r.artifact_type, r.source, r.metric_name, f"{r.metric_value:.6f}", r.repeat_index, r.raw_line, r.task_type
            ])
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "task": r.task,
                "run_id": r.run_id,
                "artifact_type": r.artifact_type,
                "source": r.source,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "repeat_index": r.repeat_index,
                "raw_line": r.raw_line,
                "stdout_snippet": r.stdout_snippet,
                "task_type": r.task_type,
            }
            for r in results
        ], f, ensure_ascii=False, indent=2)
    print(f"\n📊 Evaluation written: {csv_path}\n   JSON detail: {json_path}")
    return csv_path


def aggregate_summary(results: List[ArtifactResult], task_metadata: Dict) -> None:
    if not results:
        print("⚠️  No metrics extracted. Nothing to summarize.")
        return
    summary: Dict[str, List[float]] = {}
    task_types: Dict[str, str] = {}
    for r in results:
        key = f"{r.task}/{r.artifact_type}"
        summary.setdefault(key, []).append(r.metric_value)
        task_types[key] = r.task_type
    print("\nSUMMARY (mean metric per task/artifact_type):")
    for key, vals in sorted(summary.items()):
        mean_val = sum(vals) / len(vals)
        task_type = task_types.get(key, 'unknown')
        type_label = f" [{task_type}]" if task_type != 'unknown' else ""
        print(f"  {key}{type_label}: {mean_val:.4f} (n={len(vals)})")


def main():
    args = parse_args().parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    # Load task metadata
    repo_root = Path(__file__).parent.parent
    metadata_path = repo_root / "configs" / "task_metadata.yaml"
    task_metadata = load_task_metadata(metadata_path)
    
    manifest = read_manifest(manifest_path)
    entries = manifest.get("entries", [])
    if not entries:
        print("⚠️  Empty manifest entries")
        return
    artifact_types = set(args.artifact_types) if args.artifact_types else DEFAULT_ARTIFACT_TYPES
    task_filter = set(args.tasks) if args.tasks else None
    run_filter = set(args.runs) if args.runs else None
    print("=" * 80)
    print("EVALUATING EXTRACTED ARTIFACTS")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Artifacts considered: {', '.join(sorted(artifact_types))}")
    print(f"Repeat per artifact: {args.repeat}")
    print("=" * 80)
    results: List[ArtifactResult] = []
    for entry in entries:
        if entry["artifact_type"] not in artifact_types:
            continue
        if task_filter and entry["task"] not in task_filter:
            continue
        if run_filter and entry["run_id"] not in run_filter:
            continue
        print(f"→ {entry['task']} run {entry['run_id']} {entry['artifact_type']} :: {entry['source']}")
        try:
            evaluate_entry(entry, args, results, task_metadata)
        except Exception as e:
            print(f"   ⚠️  Evaluation error: {e}")
            if args.fail_fast:
                raise
    outdir = Path(args.output_dir)
    write_outputs(results, outdir)
    aggregate_summary(results, task_metadata)
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
