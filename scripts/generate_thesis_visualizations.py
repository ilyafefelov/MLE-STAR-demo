#!/usr/bin/env python3
"""
Generate visualizations for thesis sections 3.2 and 3.3.
Creates ablation bar charts with error bars for critical cases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Color palette
COLORS = {
    'full': '#e74c3c',           # Red - baseline
    'minimal': '#27ae60',         # Green - best
    'no_feature_engineering': '#3498db',  # Blue
    'no_tuning': '#f39c12',       # Orange
    'no_scaling': '#9b59b6',      # Purple
    'no_ensemble': '#1abc9c'      # Teal
}

CONFIG_LABELS = {
    'full': 'Full (LLM)',
    'minimal': 'Minimal',
    'no_feature_engineering': 'No FE',
    'no_tuning': 'No Tuning',
    'no_scaling': 'No Scaling',
    'no_ensemble': 'No Ensemble'
}

def load_aggregated_data():
    """Load the aggregated summary data."""
    csv_path = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/results/aggregated_summary.csv")
    df = pd.read_csv(csv_path)
    
    # Filter for N=20 runs only (latest experiments)
    df_20 = df[df['n_runs'] == 20].copy()
    
    # Extract dataset name from experiment path
    df_20['dataset'] = df_20['experiment'].apply(lambda x: x.split('/')[0])
    
    return df_20

def create_ablation_chart(data, dataset_name, title, metric_name='Score', save_path=None):
    """Create ablation bar chart for a single dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get configurations present in data
    configs = ['full', 'minimal', 'no_feature_engineering', 'no_tuning', 'no_scaling', 'no_ensemble']
    configs = [c for c in configs if c in data['configuration'].values]
    
    # Sort by mean score descending
    data_sorted = data.set_index('configuration').loc[configs].sort_values('mean', ascending=False)
    
    x = np.arange(len(data_sorted))
    width = 0.7
    
    colors = [COLORS.get(c, '#95a5a6') for c in data_sorted.index]
    labels = [CONFIG_LABELS.get(c, c) for c in data_sorted.index]
    
    # Calculate error bars (CI width)
    yerr = (data_sorted['ci_upper'] - data_sorted['ci_lower']) / 2
    
    bars = ax.bar(x, data_sorted['mean'], width, yerr=yerr, 
                  color=colors, edgecolor='black', linewidth=1,
                  capsize=5, error_kw={'capthick': 1.5, 'elinewidth': 1.5})
    
    # Find best config (highest mean)
    best_idx = data_sorted['mean'].argmax()
    full_idx = list(data_sorted.index).index('full') if 'full' in data_sorted.index else -1
    
    # Add value annotations
    for i, (bar, mean, std) in enumerate(zip(bars, data_sorted['mean'], data_sorted['std'])):
        height = bar.get_height()
        fontweight = 'bold' if i == best_idx else 'normal'
        color = 'darkgreen' if i == best_idx else 'black'
        ax.annotate(f'{mean:.3f}±{std:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + yerr.iloc[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10,
                    fontweight=fontweight, color=color)
    
    # Calculate delta from full
    if 'full' in data_sorted.index:
        full_mean = data_sorted.loc['full', 'mean']
        best_mean = data_sorted['mean'].max()
        best_config = data_sorted['mean'].idxmax()
        delta = best_mean - full_mean
        
        ax.axhline(y=full_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                   label=f'Full baseline: {full_mean:.3f}')
        
        # Add delta annotation
        if delta > 0:
            ax.text(0.98, 0.95, f'Δ = +{delta:.3f} ({CONFIG_LABELS.get(best_config, best_config)})',
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    ha='right', va='top', color='darkgreen',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel(f'{metric_name} (mean ± std)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    
    # Set y-axis limits with some padding
    y_min = max(0, data_sorted['mean'].min() - 0.15)
    y_max = min(1.0, data_sorted['mean'].max() + 0.1)
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig

def create_multi_dataset_comparison(datasets_data, title, save_path=None):
    """Create a grouped bar chart comparing multiple datasets."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_datasets = len(datasets_data)
    configs = ['full', 'minimal', 'no_feature_engineering', 'no_tuning', 'no_scaling', 'no_ensemble']
    n_configs = len(configs)
    
    width = 0.12
    x = np.arange(n_datasets)
    
    for i, config in enumerate(configs):
        means = []
        stds = []
        for dataset_name, data in datasets_data.items():
            if config in data['configuration'].values:
                row = data[data['configuration'] == config].iloc[0]
                means.append(row['mean'])
                stds.append(row['std'])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_configs/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, 
                      label=CONFIG_LABELS.get(config, config),
                      color=COLORS.get(config, '#95a5a6'),
                      edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score (mean)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(datasets_data.keys()), rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig

def create_forest_plot(summary_stats, save_path=None):
    """Create a forest plot for Cohen's d effect sizes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by effect size
    summary_stats = summary_stats.sort_values('cohens_d', ascending=True)
    
    y_pos = np.arange(len(summary_stats))
    
    # Plot effect sizes with CIs (correct column names: d_ci_lower, d_ci_upper)
    ax.errorbar(summary_stats['cohens_d'], y_pos,
                xerr=[summary_stats['cohens_d'] - summary_stats['d_ci_lower'],
                      summary_stats['d_ci_upper'] - summary_stats['cohens_d']],
                fmt='o', markersize=8, capsize=4, capthick=1.5,
                color='#2c3e50', ecolor='#7f8c8d')
    
    # Add vertical lines for effect size thresholds
    ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Small (d=0.2)')
    ax.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Medium (d=0.5)')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large (d=0.8)')
    ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, label='No effect')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_stats['dataset'])
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Forest Plot: Effect Sizes with 95% CI", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig

def main():
    """Generate all visualizations for thesis."""
    output_dir = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_aggregated_data()
    
    # Map dataset names to friendly names
    dataset_mapping = {
        'reg_california': 'California Housing (Regression)',
        'reg_synth_nonlinear': 'Synthetic Nonlinear (Regression)',
        'reg_synth_easy': 'Synthetic Easy (Regression)',
        'reg_synth_medium': 'Synthetic Medium (Regression)',
        'reg_diabetes': 'Diabetes (Regression)',
        'cls_iris': 'Iris (Classification)',
        'cls_wine': 'Wine (Classification)',
        'cls_digits': 'Digits (Classification)',
        'cls_breast_cancer': 'Breast Cancer (Classification)',
        'cls_synthetic_balanced': 'Synthetic Balanced (Classification)'
    }
    
    metric_mapping = {
        'reg_california': 'R²',
        'reg_synth_nonlinear': 'R²',
        'reg_synth_easy': 'R²',
        'reg_synth_medium': 'R²',
        'reg_diabetes': 'R²',
        'cls_iris': 'Accuracy',
        'cls_wine': 'Accuracy',
        'cls_digits': 'Accuracy',
        'cls_breast_cancer': 'Accuracy',
        'cls_synthetic_balanced': 'Accuracy'
    }
    
    print("\n=== Generating Section 3.2 Visualizations (Critical Cases) ===")
    
    # Section 3.2: Critical Over-engineering cases
    # California Housing - largest effect
    california_data = df[df['dataset'] == 'reg_california']
    if not california_data.empty:
        create_ablation_chart(
            california_data, 
            'california',
            'California Housing: Ablation Analysis (N=20)\nOver-engineering in Feature Engineering & Ensemble',
            metric_name='R²',
            save_path=output_dir / 'ablation_california_housing.png'
        )
    
    # Synthetic Nonlinear - second largest effect
    synth_nonlinear_data = df[df['dataset'] == 'reg_synth_nonlinear']
    if not synth_nonlinear_data.empty:
        create_ablation_chart(
            synth_nonlinear_data,
            'synth_nonlinear',
            'Synthetic Nonlinear: Ablation Analysis (N=20)\nDramatic Over-engineering Effect (Δ=+0.31)',
            metric_name='R²',
            save_path=output_dir / 'ablation_synth_nonlinear.png'
        )
    
    print("\n=== Generating Section 3.3 Visualizations (Critical Necessity) ===")
    
    # Section 3.3: Cases where components ARE necessary
    # Wine - scaling is critical
    wine_data = df[df['dataset'] == 'cls_wine']
    if not wine_data.empty:
        create_ablation_chart(
            wine_data,
            'wine',
            'Wine: Ablation Analysis (N=20)\nScaling is Critical (no_scaling → -26% Accuracy)',
            metric_name='Accuracy',
            save_path=output_dir / 'ablation_wine_scaling.png'
        )
    
    # Breast Cancer  
    breast_cancer_data = df[df['dataset'] == 'cls_breast_cancer']
    if not breast_cancer_data.empty:
        create_ablation_chart(
            breast_cancer_data,
            'breast_cancer',
            'Breast Cancer: Ablation Analysis (N=20)\nScaling Necessity Demonstrated',
            metric_name='Accuracy',
            save_path=output_dir / 'ablation_breast_cancer.png'
        )
    
    # Iris
    iris_data = df[df['dataset'] == 'cls_iris']
    if not iris_data.empty:
        create_ablation_chart(
            iris_data,
            'iris',
            'Iris: Ablation Analysis (N=20)\nOver-engineering in Tuning & Ensemble',
            metric_name='Accuracy',
            save_path=output_dir / 'ablation_iris.png'
        )
    
    print("\n=== Generating Summary Visualizations ===")
    
    # Multi-dataset comparison - regression
    reg_datasets = {}
    for ds in ['reg_california', 'reg_synth_easy', 'reg_synth_medium', 'reg_synth_nonlinear', 'reg_diabetes']:
        data = df[df['dataset'] == ds]
        if not data.empty:
            reg_datasets[dataset_mapping.get(ds, ds).replace(' (Regression)', '')] = data
    
    if reg_datasets:
        create_multi_dataset_comparison(
            reg_datasets,
            'Regression Datasets: Configuration Comparison (N=20)',
            save_path=output_dir / 'comparison_regression_all.png'
        )
    
    # Multi-dataset comparison - classification
    cls_datasets = {}
    for ds in ['cls_iris', 'cls_wine', 'cls_digits', 'cls_breast_cancer', 'cls_synthetic_balanced']:
        data = df[df['dataset'] == ds]
        if not data.empty:
            cls_datasets[dataset_mapping.get(ds, ds).replace(' (Classification)', '')] = data
    
    if cls_datasets:
        create_multi_dataset_comparison(
            cls_datasets,
            'Classification Datasets: Configuration Comparison (N=20)',
            save_path=output_dir / 'comparison_classification_all.png'
        )
    
    # Create forest plot from statistical analysis
    stats_path = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/results/statistical_analysis.csv")
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        create_forest_plot(
            stats_df,
            save_path=output_dir / 'forest_plot_cohens_d.png'
        )
    
    print("\n=== All visualizations generated successfully! ===")
    print(f"Output directory: {output_dir}")
    
    # List generated files
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
