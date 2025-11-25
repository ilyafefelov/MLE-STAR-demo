#!/usr/bin/env python3
"""
Generate HIGH QUALITY visualizations for thesis sections 3.2 and 3.3.
Creates publication-ready ablation bar charts with proper error bars.
Uses statistical_analysis.csv for validated N=20 data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (12, 7),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
COLORS = {
    'full': '#D55E00',           # Vermillion - baseline (LLM full)
    'minimal': '#009E73',         # Bluish green - best
    'no_fe': '#0072B2',           # Blue
    'no_tuning': '#E69F00',       # Orange
    'no_scaling': '#CC79A7',      # Reddish purple
    'no_ensemble': '#56B4E9',     # Sky blue
}

CONFIG_LABELS = {
    'full': 'Full Pipeline\n(LLM Default)',
    'minimal': 'Minimal\n(Optimized)',
    'no_fe': 'Without\nFeature Eng.',
    'no_tuning': 'Without\nHypertuning',
    'no_scaling': 'Without\nScaling',
    'no_ensemble': 'Without\nEnsemble',
}

def load_statistical_data():
    """Load the validated statistical analysis data (N=20)."""
    csv_path = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/results/statistical_analysis.csv")
    df = pd.read_csv(csv_path)
    return df

def load_raw_aggregated():
    """Load raw aggregated data for detailed ablation charts."""
    csv_path = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/results/aggregated_summary.csv")
    df = pd.read_csv(csv_path)
    # Filter for N=20 runs
    df_20 = df[df['n_runs'] == 20].copy()
    df_20['dataset'] = df_20['experiment'].apply(lambda x: x.split('/')[0])
    return df_20

def create_comparison_chart_hq(stats_df, task_type, output_path):
    """
    Create high-quality comparison bar chart showing best vs full pipeline.
    
    Parameters:
    - stats_df: DataFrame with statistical analysis
    - task_type: 'regression' or 'classification'
    - output_path: Path to save figure
    """
    # Filter by task type
    if task_type == 'regression':
        data = stats_df[stats_df['dataset'].str.startswith('reg_')].copy()
        metric_label = 'R² Score'
        title = 'Regression Tasks: Optimal vs Full Pipeline Performance'
    else:
        data = stats_df[stats_df['dataset'].str.startswith('cls_')].copy()
        metric_label = 'Accuracy'
        title = 'Classification Tasks: Optimal vs Full Pipeline Performance'
    
    # Sort by delta (impact of simplification)
    data = data.sort_values('delta', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(data))
    width = 0.35
    
    # Bars for best config and full pipeline
    bars1 = ax.bar(x - width/2, data['best_score'], width, 
                   label='Optimal Configuration', color=COLORS['minimal'],
                   edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, data['full_score'], width,
                   label='Full Pipeline (LLM)', color=COLORS['full'],
                   edgecolor='black', linewidth=1)
    
    # Error bars
    ax.errorbar(x - width/2, data['best_score'], yerr=1.96*data['best_std'],
                fmt='none', color='black', capsize=4, capthick=1.5)
    ax.errorbar(x + width/2, data['full_score'], yerr=1.96*data['full_std'],
                fmt='none', color='black', capsize=4, capthick=1.5)
    
    # Labels and formatting
    dataset_labels = [d.replace('reg_', '').replace('cls_', '').replace('_', ' ').title() 
                      for d in data['dataset']]
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=15, ha='right')
    ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.95)
    
    # Add delta annotations
    for i, (idx, row) in enumerate(data.iterrows()):
        delta = row['delta']
        significance = row['p_interpret']
        y_pos = max(row['best_score'], row['full_score']) + 0.02
        
        if significance == '***':
            text = f'Δ = +{delta:.3f}***'
            color = '#009E73'
        elif significance == '**':
            text = f'Δ = +{delta:.3f}**'
            color = '#0072B2'
        elif significance == '*':
            text = f'Δ = +{delta:.3f}*'
            color = '#E69F00'
        else:
            text = f'Δ = {delta:+.3f} (ns)'
            color = 'gray'
        
        ax.annotate(text, xy=(i, y_pos), ha='center', fontsize=10, 
                    fontweight='bold', color=color)
    
    # Add effect size info
    info_text = "Effect sizes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            style='italic', color='gray')
    
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def create_forest_plot_hq(stats_df, output_path):
    """Create high-quality forest plot for Cohen's d effect sizes."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort by effect size
    data = stats_df.sort_values('cohens_d', ascending=True).copy()
    y_pos = np.arange(len(data))
    
    # Color by significance
    colors = []
    for _, row in data.iterrows():
        if row['p_interpret'] == '***':
            colors.append('#009E73')  # Green - highly significant
        elif row['p_interpret'] in ['**', '*']:
            colors.append('#E69F00')  # Orange - significant
        else:
            colors.append('#999999')  # Gray - not significant
    
    # Plot effect sizes with CIs
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.errorbar(row['cohens_d'], i,
                    xerr=[[row['cohens_d'] - row['d_ci_lower']], 
                          [row['d_ci_upper'] - row['cohens_d']]],
                    fmt='o', markersize=10, capsize=6, capthick=2,
                    color=colors[i], ecolor=colors[i], elinewidth=2,
                    markeredgecolor='black', markeredgewidth=1)
    
    # Effect size threshold lines
    ax.axvline(x=0, color='#D55E00', linestyle='-', linewidth=2, alpha=0.7, label='No Effect')
    ax.axvline(x=0.2, color='#E69F00', linestyle='--', linewidth=1.5, alpha=0.7, label='Small (d=0.2)')
    ax.axvline(x=0.5, color='#0072B2', linestyle='--', linewidth=1.5, alpha=0.7, label='Medium (d=0.5)')
    ax.axvline(x=0.8, color='#009E73', linestyle='--', linewidth=1.5, alpha=0.7, label='Large (d=0.8)')
    
    # Dataset labels
    dataset_labels = [d.replace('reg_', '').replace('cls_', '').replace('_', ' ').title() 
                      for d in data['dataset']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dataset_labels, fontsize=12)
    
    # Add Cohen's d values on right side
    for i, (idx, row) in enumerate(data.iterrows()):
        d_text = f"d = {row['cohens_d']:.2f} [{row['d_ci_lower']:.2f}, {row['d_ci_upper']:.2f}]"
        ax.annotate(d_text, xy=(max(row['d_ci_upper'] + 0.5, 2), i),
                    fontsize=10, va='center', color=colors[i], fontweight='bold')
    
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
    ax.set_title("Forest Plot: Effect Sizes with 95% Confidence Intervals\nOptimal vs Full Pipeline (N=20 per dataset)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_xlim(-1, 20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def create_critical_case_chart(dataset_name, display_name, output_path, stats_df, agg_df):
    """
    Create detailed ablation chart for critical cases.
    Shows all configurations with proper error bars.
    """
    # Get data for this dataset
    ds_data = agg_df[agg_df['dataset'] == dataset_name].copy()
    
    if len(ds_data) == 0:
        print(f"Warning: No data for {dataset_name}")
        return
    
    # Aggregate by configuration
    config_stats = ds_data.groupby('configuration').agg({
        'mean': 'mean',
        'std': 'mean',
        'ci_lower': 'mean',
        'ci_upper': 'mean'
    }).reset_index()
    
    # Define order and filter
    config_order = ['minimal', 'no_feature_engineering', 'no_ensemble', 'no_tuning', 'full', 'no_scaling']
    config_stats = config_stats[config_stats['configuration'].isin(config_order)]
    config_stats['order'] = config_stats['configuration'].apply(lambda x: config_order.index(x) if x in config_order else 99)
    config_stats = config_stats.sort_values('order')
    
    # Map no_feature_engineering to no_fe for colors
    def get_color(cfg):
        if cfg == 'no_feature_engineering':
            return COLORS['no_fe']
        return COLORS.get(cfg, '#95a5a6')
    
    def get_label(cfg):
        if cfg == 'no_feature_engineering':
            return CONFIG_LABELS['no_fe']
        return CONFIG_LABELS.get(cfg, cfg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(config_stats))
    width = 0.7
    
    colors = [get_color(c) for c in config_stats['configuration']]
    labels = [get_label(c) for c in config_stats['configuration']]
    
    # Error bars (95% CI)
    yerr_lower = config_stats['mean'] - config_stats['ci_lower']
    yerr_upper = config_stats['ci_upper'] - config_stats['mean']
    
    # Handle negative or invalid CIs
    yerr_lower = yerr_lower.clip(lower=0)
    yerr_upper = yerr_upper.clip(lower=0)
    
    bars = ax.bar(x, config_stats['mean'], width, 
                  yerr=[yerr_lower, yerr_upper],
                  color=colors, edgecolor='black', linewidth=1.5,
                  capsize=6, error_kw={'capthick': 2, 'elinewidth': 2})
    
    # Find best and worst
    best_idx = config_stats['mean'].idxmax()
    best_pos = config_stats.index.get_loc(best_idx)
    
    # Add value annotations
    for i, (idx, row) in enumerate(config_stats.iterrows()):
        mean_val = row['mean']
        std_val = row['std']
        y_pos = mean_val + yerr_upper.iloc[i] + 0.02
        
        # Highlight best config
        if i == best_pos:
            fontweight = 'bold'
            color = '#009E73'
            text = f'{mean_val:.3f}±{std_val:.3f}\n(Best)'
        else:
            fontweight = 'normal'
            color = 'black'
            text = f'{mean_val:.3f}±{std_val:.3f}'
        
        ax.annotate(text, xy=(i, y_pos), ha='center', va='bottom',
                    fontsize=11, fontweight=fontweight, color=color)
    
    # Get statistical info
    stat_row = stats_df[stats_df['dataset'] == dataset_name]
    if len(stat_row) > 0:
        stat_row = stat_row.iloc[0]
        delta = stat_row['delta']
        d = stat_row['cohens_d']
        p = stat_row['p_interpret']
        subtitle = f"Δ = {delta:+.3f}, Cohen's d = {d:.2f}, p{p}"
    else:
        subtitle = ""
    
    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    
    metric = 'R²' if dataset_name.startswith('reg_') else 'Accuracy'
    ax.set_ylabel(f'{metric} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{display_name}: Ablation Study Results\n{subtitle}', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Add N=20 annotation
    ax.text(0.98, 0.02, 'N = 20 runs per configuration\nError bars: 95% CI',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set y limits
    y_max = config_stats['mean'].max() + yerr_upper.max() + 0.1
    ax.set_ylim(0, min(y_max, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Generate all high-quality thesis visualizations."""
    output_dir = Path("D:/School/GoIT/MAUP_REDO_HWs/Diploma/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    stats_df = load_statistical_data()
    agg_df = load_raw_aggregated()
    
    print("\n=== Generating Section 3.2: Critical Over-engineering Cases ===")
    # California Housing - huge delta with minimal config
    create_critical_case_chart(
        'reg_california', 
        'California Housing (Regression)',
        output_dir / 'fig_3_2_california.png',
        stats_df, agg_df
    )
    
    # Synth Nonlinear - large delta
    create_critical_case_chart(
        'reg_synth_nonlinear',
        'Synthetic Nonlinear (Regression)',
        output_dir / 'fig_3_2_synth_nonlinear.png',
        stats_df, agg_df
    )
    
    print("\n=== Generating Section 3.3: Critical Necessity Cases ===")
    # Iris - FE is critical
    create_critical_case_chart(
        'cls_iris',
        'Iris (Classification)',
        output_dir / 'fig_3_3_iris.png',
        stats_df, agg_df
    )
    
    # Breast Cancer
    create_critical_case_chart(
        'cls_breast_cancer',
        'Breast Cancer (Classification)',
        output_dir / 'fig_3_3_breast_cancer.png',
        stats_df, agg_df
    )
    
    # Wine
    create_critical_case_chart(
        'cls_wine',
        'Wine (Classification)',
        output_dir / 'fig_3_3_wine.png',
        stats_df, agg_df
    )
    
    print("\n=== Generating Summary Visualizations ===")
    # Regression comparison
    create_comparison_chart_hq(
        stats_df, 'regression',
        output_dir / 'fig_summary_regression.png'
    )
    
    # Classification comparison
    create_comparison_chart_hq(
        stats_df, 'classification',
        output_dir / 'fig_summary_classification.png'
    )
    
    # Forest plot
    create_forest_plot_hq(
        stats_df,
        output_dir / 'fig_forest_plot.png'
    )
    
    print("\n=== All high-quality visualizations generated! ===")
    print(f"Output directory: {output_dir}")
    
    # List generated files
    for f in sorted(output_dir.glob('fig_*.png')):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
