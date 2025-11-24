#!/usr/bin/env python
"""
Generate a global report summarizing all experiment results.
"""

import os
import pandas as pd
from pathlib import Path
import glob
import yaml

def get_latest_summary_file(experiment_dir):
    """Find the latest summary_statistics_*.csv file in the experiment directory."""
    # The structure is results/<exp_name>/<wrapper_name>/summary_statistics_*.csv
    # We need to search recursively or assume the structure.
    
    # Find all summary files
    files = list(Path(experiment_dir).rglob("summary_statistics_*.csv"))
    if not files:
        return None
    
    # Sort by modification time (or filename timestamp)
    # Filenames have timestamps, so sorting by name is also effective if format is consistent
    files.sort(key=lambda x: x.name, reverse=True)
    return files[0]

def main():
    results_dir = Path("results")
    config_path = Path("configs/experiment_suite.yaml")
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiments = config.get('experiments', [])
    
    report_data = []
    
    print(f"Scanning results in {results_dir}...")
    
    for exp in experiments:
        exp_name = exp['name']
        dataset = exp['dataset']
        task_type = exp['task_type']
        
        exp_dir = results_dir / exp_name
        if not exp_dir.exists():
            print(f"  [MISSING] {exp_name} (Directory not found)")
            continue
            
        summary_file = get_latest_summary_file(exp_dir)
        if not summary_file:
            print(f"  [MISSING] {exp_name} (No summary file found)")
            continue
            
        print(f"  [FOUND] {exp_name} -> {summary_file.name}")
        
        try:
            df = pd.read_csv(summary_file)
            # Get the best configuration (first row usually, or sort by mean)
            # The summary files are usually sorted by mean descending (for accuracy/r2)
            # But let's double check.
            
            # Identify metric column
            metric_col = 'mean' # The summary file has 'mean', 'std', etc.
            # But what is the metric name? It's not in the CSV columns usually, 
            # but the values tell us.
            # We can assume the first row is the "best" as per the ablation script logic.
            
            best_row = df.iloc[0]
            baseline_row = df[df['configuration'] == 'baseline'].iloc[0] if 'baseline' in df['configuration'].values else None
            full_row = df[df['configuration'] == 'full'].iloc[0] if 'full' in df['configuration'].values else None
            
            best_config = best_row['configuration']
            best_score = best_row['mean']
            best_std = best_row['std']
            
            baseline_score = baseline_row['mean'] if baseline_row is not None else float('nan')
            full_score = full_row['mean'] if full_row is not None else float('nan')
            
            report_data.append({
                'Experiment': exp_name,
                'Dataset': dataset,
                'Task': task_type,
                'Best Config': best_config,
                'Best Score': best_score,
                'Std Dev': best_std,
                'Baseline Score': baseline_score,
                'Full Pipeline Score': full_score,
                'Improvement over Baseline': best_score - baseline_score,
                'Improvement over Full': best_score - full_score
            })
            
        except Exception as e:
            print(f"  [ERROR] Could not process {summary_file}: {e}")

    if not report_data:
        print("No results found.")
        return

    report_df = pd.DataFrame(report_data)
    
    # Format the report
    print("\n" + "="*80)
    print("GLOBAL EXPERIMENT REPORT")
    print("="*80)
    
    # Split by task type
    for task in report_df['Task'].unique():
        print(f"\nTask Type: {task.upper()}")
        task_df = report_df[report_df['Task'] == task]
        
        # Select and rename columns for display
        display_cols = ['Experiment', 'Dataset', 'Best Config', 'Best Score', 'Baseline Score', 'Full Pipeline Score']
        print(task_df[display_cols].to_markdown(index=False, floatfmt=".4f"))

    # Save to file
    output_file = "docs/GLOBAL_EXPERIMENT_REPORT.md"
    with open(output_file, "w") as f:
        f.write("# Global Experiment Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        for task in report_df['Task'].unique():
            f.write(f"## Task Type: {task.upper()}\n\n")
            task_df = report_df[report_df['Task'] == task]
            f.write(task_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            
        f.write("## Analysis Notes\n\n")
        f.write("- **Best Score**: The mean metric (Accuracy for classification, R2 for regression) of the best performing configuration.\n")
        f.write("- **Baseline**: Dummy classifier/regressor performance.\n")
        f.write("- **Full Pipeline**: The performance of the complete Gemini-generated pipeline.\n")
        
    print(f"\nReport saved to {output_file}")

if __name__ == "__main__":
    main()
