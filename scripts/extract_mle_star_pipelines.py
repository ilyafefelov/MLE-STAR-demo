#!/usr/bin/env python
"""Extract and consolidate MLE-STAR generated pipeline artifacts.

This script scans the MLE-STAR workspace produced by the ADK run command and
copies any Python pipeline/code artifacts into our project's `generated_pipelines` folder.

Features:
1. Detect tasks and run IDs automatically (folders: workspace/<task>/<run_id>/)
2. Copy python files (init_code_*.py, train*.py, refined*.py, ensemble*.py, *.py) with
   a normalized filename: <task>_run<id>_<original>.
3. (Optional) Extract model candidate descriptions from `model_candidates/`.
4. Create a manifest JSON with metadata (source path, destination path, file type).
5. Dry-run mode to preview actions.

Usage examples:
    python scripts/extract_mle_star_pipelines.py                # extract all
    python scripts/extract_mle_star_pipelines.py --tasks iris_gbdt wine_gbdt
    python scripts/extract_mle_star_pipelines.py --runs 2 3      # only runs 2 & 3
    python scripts/extract_mle_star_pipelines.py --include-candidates
    python scripts/extract_mle_star_pipelines.py --dry-run

Author: Фефелов Ілля Олександрович
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import importlib.util
import sys
from src.mle_star_ablation import mle_star_generated_pipeline as mgp

# Root to ADK MLE-STAR workspace (relative to repo root)
WORKSPACE_ROOT = (
    Path(__file__).parent.parent
    / "adk-samples" / "python" / "agents" / "machine-learning-engineering"
    / "machine_learning_engineering" / "workspace"
)

DEFAULT_OUTPUT_DIR = Path("generated_pipelines")

PY_FILE_PATTERN = re.compile(r".*\.py$")


def discover_tasks(workspace_root: Path) -> List[Path]:
    """Return list of task directories in workspace root."""
    if not workspace_root.exists():
        return []
    return [p for p in workspace_root.iterdir() if p.is_dir()]


def discover_runs(task_dir: Path) -> List[Path]:
    """Return sorted list of run directories (numeric names)."""
    runs = []
    for p in task_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            runs.append(p)
    return sorted(runs, key=lambda r: int(r.name))


def classify_file(path: Path) -> str:
    """Infer artifact type by filename."""
    name = path.name
    if name.startswith("init_code_"):
        return "initialization"
    if name.startswith("train"):
        return "train"
    if name.startswith("refined"):
        return "refinement"
    if name.startswith("ensemble"):
        return "ensemble"
    if name.startswith("pipeline"):
        return "pipeline"
    return "other"


def extract_run(task: str, run_dir: Path, outdir: Path, include_candidates: bool, dry_run: bool) -> List[Dict]:
    """Extract all python files (and optionally model candidates) for a run.

    Returns list of manifest entries.
    """
    manifest: List[Dict] = []

    for py_file in run_dir.glob("*.py"):
        if not PY_FILE_PATTERN.match(py_file.name):
            continue
        artifact_type = classify_file(py_file)
        dest_name = f"{task}_run{run_dir.name}_{py_file.name}"
        dest_path = outdir / dest_name

        entry = {
            "task": task,
            "run_id": run_dir.name,
            "source": str(py_file),
            "destination": str(dest_path),
            "artifact_type": artifact_type,
        }
        manifest.append(entry)

        if dry_run:
            print(f"📝 [dry-run] Would copy: {py_file} -> {dest_path}")
        else:
            # Avoid overwriting: if exists, add numeric suffix
            final_dest = dest_path
            counter = 1
            while final_dest.exists():
                final_dest = dest_path.with_name(dest_path.stem + f"_{counter}" + dest_path.suffix)
                counter += 1
            shutil.copy2(py_file, final_dest)
            entry["destination"] = str(final_dest)
            print(f"✅ Copied {py_file.name} -> {final_dest.name}")
            # Try to import and inspect the copied file to ensure it contains expected components
            try:
                spec = importlib.util.spec_from_file_location(final_dest.stem, str(final_dest))
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                # Find a possible builder function
                candidates = ['build_full_pipeline', 'create_model_pipeline', 'create_pipeline']
                builder = None
                for cand in candidates:
                    if hasattr(module, cand):
                        builder = getattr(module, cand)
                        break
                if builder is not None:
                    try:
                        pipeline = None
                        try:
                            pipeline = builder(random_state=42)
                        except TypeError:
                            pipeline = builder()
                        pipeline = mgp._ensure_pipeline_object(pipeline)
                        info = mgp.inspect_pipeline(pipeline)
                        print(f"   🔎 Inspect: steps={info['steps']}, has_scaler={info['has_scaler']}, has_feature_engineering={info['has_feature_engineering']}, has_tuning={info['has_tuning']}, has_ensembling={info['has_ensembling']}")
                        if not mgp.is_ablation_meaningful(pipeline):
                            print(f"   ⚠️  Warning: Pipeline seems to lack enough ablation-sensitive components (scaler/feature_engineering/tuning/ensembling). Consider regenerating with a stronger prompt.")
                    except Exception as e:
                        print(f"   ⚠️  Inspect failed for {final_dest.name}: {e}")
                else:
                    print(f"   ⚠️  No builder found in {final_dest.name}; skipping inspection.")
            except Exception as e:
                print(f"   ⚠️  Could not import/inspect copied file {final_dest.name}: {e}")

    if include_candidates:
        candidates_dir = run_dir / "model_candidates"
        if candidates_dir.exists():
            for txt in candidates_dir.glob("*.txt"):
                dest_name = f"{task}_run{run_dir.name}_{txt.name}"
                dest_path = outdir / dest_name
                entry = {
                    "task": task,
                    "run_id": run_dir.name,
                    "source": str(txt),
                    "destination": str(dest_path),
                    "artifact_type": "candidate_description",
                }
                manifest.append(entry)
                if dry_run:
                    print(f"📝 [dry-run] Would copy candidate: {txt} -> {dest_path}")
                else:
                    shutil.copy2(txt, dest_path)
                    print(f"📄 Candidate {txt.name} saved")

    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MLE-STAR generated pipeline files from workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Filter by task names (default: all tasks)")
    parser.add_argument("--runs", nargs="*", default=None, help="Filter by run IDs (numeric) e.g. 1 2 3")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="Destination directory for extracted files")
    parser.add_argument("--include-candidates", action="store_true", help="Also copy model candidate description .txt files")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without copying files")
    parser.add_argument("--manifest", default="mle_star_extraction_manifest.json", help="Filename for manifest JSON")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("MLE-STAR PIPELINE EXTRACTION")
    print("=" * 80)
    print(f"Workspace root: {WORKSPACE_ROOT}")
    print(f"Output dir: {args.outdir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Include candidates: {args.include_candidates}")
    print("=" * 80)

    if not WORKSPACE_ROOT.exists():
        print(f"❌ Workspace root not found: {WORKSPACE_ROOT}")
        return

    outdir = Path(args.outdir)
    if not args.dry_run:
        outdir.mkdir(exist_ok=True, parents=True)

    requested_tasks = set(args.tasks) if args.tasks else None
    requested_runs = set(args.runs) if args.runs else None

    tasks = discover_tasks(WORKSPACE_ROOT)
    if not tasks:
        print("⚠️  No task directories found.")
        return

    manifest: List[Dict] = []
    total_files = 0

    for task_dir in tasks:
        task_name = task_dir.name
        if requested_tasks and task_name not in requested_tasks:
            continue
        runs = discover_runs(task_dir)
        if not runs:
            print(f"⚠️  {task_name}: no runs found")
            continue
        print(f"\n🔍 Task: {task_name} (runs: {', '.join(r.name for r in runs)})")
        for run_dir in runs:
            if requested_runs and run_dir.name not in requested_runs:
                continue
            print(f"  → Extracting run {run_dir.name}")
            run_manifest = extract_run(
                task=task_name,
                run_dir=run_dir,
                outdir=outdir,
                include_candidates=args.include_candidates,
                dry_run=args.dry_run,
            )
            manifest.extend(run_manifest)
            total_files += len(run_manifest)

    # Write manifest
    if not args.dry_run:
        manifest_path = outdir / args.manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "workspace_root": str(WORKSPACE_ROOT),
                    "total_entries": total_files,
                    "entries": manifest,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n🧾 Manifest saved: {manifest_path} ({total_files} entries)")
    else:
        print(f"\n🧪 Dry run complete. Would extract {total_files} artifacts.")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
