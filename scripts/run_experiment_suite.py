#!/usr/bin/env python
"""Coordinate multiple ablation runs based on a YAML manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml


def _format_cmd(cmd: List[str]) -> str:
    """Return a shell-friendly representation for logs."""
    try:
        import shlex

        return " ".join(shlex.quote(part) for part in cmd)
    except Exception:
        return " ".join(cmd)


def _bool(value, fallback=False):
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_command(exp: dict, global_cfg: dict, python_bin: str) -> List[str]:
    output_base = Path(global_cfg.get("output_dir", "results"))
    exp_output = output_base / exp["name"]
    exp_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        "scripts/run_ablation.py",
        "--dataset",
        exp["dataset"],
        "--output-dir",
        str(exp_output),
        "--n-runs",
        str(exp.get("n_runs", global_cfg.get("n_runs", 3))),
    ]

    if exp.get("csv_path"):
        cmd.extend(["--csv-path", exp["csv_path"]])
    if exp.get("target"):
        cmd.extend(["--target", exp["target"]])
    if exp.get("pipeline_file"):
        cmd.extend(["--pipeline-file", exp["pipeline_file"]])
    if exp.get("variant"):
        cmd.extend(["--variant", exp["variant"]])
    if exp.get("task_type"):
        cmd.extend(["--task-type", exp["task_type"]])

    seed = exp.get("seed", global_cfg.get("base_seed"))
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    deterministic = exp.get("deterministic")
    if deterministic is None:
        deterministic = global_cfg.get("deterministic", False)
    if _bool(deterministic):
        cmd.append("--deterministic")

    extra_args = exp.get("extra_args", []) or []
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Batch runner for experiment manifests")
    parser.add_argument(
        "--config",
        default="configs/experiment_suite.yaml",
        help="Path to YAML manifest",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands instead of dry-run printing",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to include",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Experiment names to skip even if enabled",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Manifest not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    global_cfg = data.get("global", {})
    experiments = data.get("experiments", [])

    if not experiments:
        raise SystemExit("No experiments defined in manifest")

    if args.only:
        only_set = set(args.only)
    else:
        only_set = None
    skip_set = set(args.skip or [])

    python_bin = global_cfg.get("python") or sys.executable
    dry_run = not args.execute

    planned = []
    for exp in experiments:
        name = exp.get("name")
        if not name:
            continue
        if only_set and name not in only_set:
            continue
        if name in skip_set:
            continue
        if not _bool(exp.get("enabled", False)):
            continue
        planned.append(exp)

    if not planned:
        print("No experiments enabled after filters.")
        return

    print(f"Planning {len(planned)} experiment(s) from {config_path} (dry_run={dry_run})")

    for idx, exp in enumerate(planned, start=1):
        cmd = build_command(exp, global_cfg, python_bin)
        print(f"[{idx}/{len(planned)}] {exp['name']}: {_format_cmd(cmd)}")

        if dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Experiment {exp['name']} failed with exit code {result.returncode}")

    if dry_run:
        print("Dry run complete. Re-run with --execute to launch experiments.")
    else:
        print("All experiments executed successfully.")


if __name__ == "__main__":
    main()
