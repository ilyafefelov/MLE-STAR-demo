#!/usr/bin/env python
"""
Master Experiment Orchestrator: ADK Agent vs. Single-Shot Gemini.

This script runs a complete comparative study:
1. Runs ADK MSE-STAR Agent (Agentic Workflow).
2. Runs Single-Shot Gemini Generation (Direct API).
3. Runs Ablation Studies on artifacts from BOTH methods.
4. Aggregates and visualizes the results.

Usage:
  python scripts/run_master_experiment.py --tasks iris --n-runs 3
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCRIPTS_DIR = Path("scripts")
RESULTS_DIR = Path("results")
MODEL_COMP_DIR = Path("model_comparison_results")
GENERATED_DIR = Path("generated_pipelines")

def load_env():
    env_path = Path('.env')
    if env_path.exists():
        print(f"Loading environment from {env_path}")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    k, v = line.strip().split('=', 1)
                    os.environ.setdefault(k, v)
    
    # Map GOOGLE_API_KEY to GEMINI_API_KEY if needed
    if 'GEMINI_API_KEY' not in os.environ:
        if 'GOOGLE_API_KEY' in os.environ:
            os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY']
            print("Mapped GOOGLE_API_KEY to GEMINI_API_KEY")
        elif 'GEMINI_API_TOKEN' in os.environ:
            os.environ['GEMINI_API_KEY'] = os.environ['GEMINI_API_TOKEN']
            print("Mapped GEMINI_API_TOKEN to GEMINI_API_KEY")

    if 'GEMINI_API_KEY' in os.environ:
        key = os.environ['GEMINI_API_KEY']
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        print(f"GEMINI_API_KEY loaded: {masked}")
    else:
        print("⚠️ GEMINI_API_KEY not found in environment or .env")

def run_command(cmd, description):
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*80}")
    start = time.time()
    proc = subprocess.run(cmd, text=True)
    duration = time.time() - start
    if proc.returncode != 0:
        print(f"❌ Failed (exit {proc.returncode}) after {duration:.1f}s")
        # We don't exit immediately to allow partial results to be processed
    else:
        print(f"✅ Success ({duration:.1f}s)")
    return proc.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["iris"], help="Datasets to run (iris, wine, breast_cancer, digits)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of ablation runs per pipeline")
    parser.add_argument("--skip-adk-gen", action="store_true", help="Skip ADK generation (use existing workspace)")
    parser.add_argument("--skip-single-shot-gen", action="store_true", help="Skip Single-Shot generation (use existing files)")
    args = parser.parse_args()

    load_env()
    tasks = args.tasks
    
    # --- PHASE 1: ADK AGENT ---
    if not args.skip_adk_gen:
        # 1. Run Agent
        run_command(
            [sys.executable, str(SCRIPTS_DIR / "run_mle_star_batch.py"), "--tasks"] + tasks,
            "Running ADK MSE-STAR Agent"
        )
        # 2. Extract
        run_command(
            [sys.executable, str(SCRIPTS_DIR / "extract_mle_star_pipelines.py"), "--tasks"] + tasks,
            "Extracting ADK Pipelines"
        )

    # 3. Wrap & Ablate ADK Artifacts
    print("\n🔍 Processing ADK Artifacts for Ablation...")
    adk_wrappers = []
    for task in tasks:
        # Find extracted training/refined scripts
        # We look for files that start with the task name and contain 'train' or 'refined'
        candidates = list(GENERATED_DIR.glob(f"{task}_run*_train*.py")) + \
                     list(GENERATED_DIR.glob(f"{task}_run*_refined*.py"))
        
        if not candidates:
            print(f"⚠️ No ADK candidates found for task: {task}")

        for script in candidates:
            wrapper_name = f"adk_wrapper_{script.stem}.py"
            wrapper_path = MODEL_COMP_DIR / wrapper_name
            
            # Create Wrapper
            ret = run_command(
                [sys.executable, str(SCRIPTS_DIR / "auto_make_pipeline_wrapper.py"), 
                 "--src", str(script), "--out", str(wrapper_path)],
                f"Creating wrapper for {script.name}"
            )
            
            if ret == 0:
                adk_wrappers.append((task, wrapper_path))

    # Run Ablation on ADK Wrappers
    for task, wrapper in adk_wrappers:
        output_subdir = RESULTS_DIR / f"adk_{wrapper.stem}"
        run_command(
            [sys.executable, str(SCRIPTS_DIR / "run_ablation.py"), 
             "--dataset", task, 
             "--pipeline-file", str(wrapper),
             "--n-runs", str(args.n_runs),
             "--output-dir", str(output_subdir)],
            f"Ablation Study: ADK {wrapper.stem}"
        )

    # --- PHASE 2: SINGLE SHOT ---
    if not args.skip_single_shot_gen:
        # 1. Generate Single Shot
        cmd = [sys.executable, str(SCRIPTS_DIR / "compare_gemini_models_on_datasets.py"), "--datasets"] + tasks
        if 'GEMINI_API_KEY' in os.environ:
             cmd.extend(["--api-key", os.environ['GEMINI_API_KEY']])
        
        run_command(cmd, "Generating Single-Shot Pipelines")

    # 2. Ablate Single Shot
    # Find generated single-shot files (pattern: gemini_*.py, excluding wrappers)
    print("\n🔍 Processing Single-Shot Artifacts for Ablation...")
    for task in tasks:
        # Look for standard single-shot naming patterns
        # compare_gemini_models_on_datasets.py produces: gemini_2.5_flash_lite_iris.py etc.
        candidates = list(MODEL_COMP_DIR.glob(f"gemini_*_{task}.py"))
        
        for script in candidates:
            if "wrapper" in script.name: continue # Skip existing wrappers
            
            output_subdir = RESULTS_DIR / f"single_shot_{script.stem}"
            run_command(
                [sys.executable, str(SCRIPTS_DIR / "run_ablation.py"),
                 "--dataset", task,
                 "--pipeline-file", str(script),
                 "--n-runs", str(args.n_runs),
                 "--output-dir", str(output_subdir)],
                f"Ablation Study: Single-Shot {script.stem}"
            )

    # --- PHASE 3: AGGREGATE & VISUALIZE ---
    run_command(
        [sys.executable, str(SCRIPTS_DIR / "aggregate_results.py"), "--results-dir", str(RESULTS_DIR)],
        "Aggregating Results"
    )
    
    # Simple Visualization of the Aggregate
    agg_file = Path("reports/aggregate_summary_stats.csv")
    if agg_file.exists():
        print("\n📊 Generating Comparison Plot...")
        try:
            df = pd.read_csv(agg_file)
            
            def classify_source(path_str):
                if "adk" in str(path_str): return "ADK Agent"
                if "single_shot" in str(path_str) or "gemini" in str(path_str): return "Single Shot"
                return "Unknown"

            df['Method'] = df['path'].apply(classify_source)
            
            # Filter for 'standard' configuration to compare baseline performance
            # or aggregate across all configs
            subset = df[df['configuration'] == 'standard']
            if subset.empty:
                subset = df # Fallback if standard not found
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=subset, x='dataset_variant', y='mean', hue='Method')
            plt.title("ADK Agent vs Single-Shot: Performance Comparison (Standard Config)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("reports/master_comparison.png")
            print("✅ Saved plot to reports/master_comparison.png")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")

if __name__ == "__main__":
    main()
