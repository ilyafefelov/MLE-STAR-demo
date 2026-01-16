#!/usr/bin/env python3
"""
Enhanced Analysis Script
========================
Generates:
1. Cohen's d effect sizes with classification (large/medium/small/negligible)
2. Forest Plot visualization of all datasets
3. Updated summary table for the report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
REPORTS_DIR = Path(__file__).parent.parent / "reports"
OUTPUT_DIR = REPORTS_DIR


def cohens_d(mean1, mean2, std1, std2, n1=20, n2=20):
    """Calculate Cohen's d effect size."""
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def classify_effect(d):
    """Classify Cohen's d effect size."""
    d = abs(d)
    if d >= 0.8:
        return "Large"
    elif d >= 0.5:
        return "Medium"
    elif d >= 0.2:
        return "Small"
    else:
        return "Negligible"


def load_and_process_data():
    """Load aggregate stats and extract best N=20 results per dataset."""
    csv_path = REPORTS_DIR / "aggregate_summary_stats.csv"
    df = pd.read_csv(csv_path)
    
    # Filter to N=20 runs only (most robust results)
    df_n20 = df[df['n_runs'] == 20].copy()
    
    # Use dataset_variant as unique identifier
    df_n20['dataset_clean'] = df_n20['dataset_variant']
    
    # Get unique datasets
    datasets = df_n20['dataset_clean'].unique()
    
    results = []
    
    for ds in datasets:
        ds_data = df_n20[df_n20['dataset_clean'] == ds]
        
        # Get full pipeline stats
        full_stats = ds_data[ds_data['configuration'] == 'full']
        if full_stats.empty:
            continue
        full_stats = full_stats.iloc[0]
        
        # Get best alternative (minimal or no_feature_engineering)
        alternatives = ds_data[ds_data['configuration'].isin(['minimal', 'no_feature_engineering'])]
        if alternatives.empty:
            continue
        best_alt = alternatives.loc[alternatives['mean'].idxmax()]
        
        # Calculate effect size
        d = cohens_d(
            best_alt['mean'], full_stats['mean'],
            best_alt['std'], full_stats['std']
        )
        
        results.append({
            'Dataset': ds,
            'Best Config': best_alt['configuration'],
            'Best Score': best_alt['mean'],
            'Best CI Lower': best_alt['ci_lower'],
            'Best CI Upper': best_alt['ci_upper'],
            'Best Std': best_alt['std'],
            'Full Score': full_stats['mean'],
            'Full CI Lower': full_stats['ci_lower'],
            'Full CI Upper': full_stats['ci_upper'],
            'Full Std': full_stats['std'],
            'Delta': best_alt['mean'] - full_stats['mean'],
            'Cohen_d': d,
            'Effect Size': classify_effect(d)
        })
    
    return pd.DataFrame(results)


def create_forest_plot(df):
    """Create a forest plot showing effect sizes across all datasets."""
    # Sort by effect size
    df_sorted = df.sort_values('Delta', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_positions = range(len(df_sorted))
    
    # Mapping effect sizes to Ukrainian
    effect_ua = {
        'Large': 'Великий',
        'Medium': 'Середній',
        'Small': 'Малий',
        'Negligible': 'Незначний'
    }
    
    # Colors based on effect size
    colors = {
        'Large': '#2ecc71',      # Green
        'Medium': '#f39c12',     # Orange
        'Small': '#3498db',      # Blue
        'Negligible': '#95a5a6'  # Gray
    }
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = colors.get(row['Effect Size'], '#95a5a6')
        
        # Plot point for best config
        ax.scatter(row['Delta'], i, color=color, s=150, zorder=3, marker='o')
        
        # Error bar based on CI
        ci_width = (row['Best CI Upper'] - row['Best CI Lower']) / 2
        ax.errorbar(row['Delta'], i, xerr=ci_width, color=color, capsize=5, capthick=2, linewidth=2, zorder=2)
    
    # Reference line at 0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Без різниці')
    
    # Labels
    ax.set_yticks(y_positions)
    labels = [f"{row['Dataset']}\n({row['Best Config']})\nd={row['Cohen_d']:.2f} [{effect_ua.get(row['Effect Size'], row['Effect Size'])}]" 
              for _, row in df_sorted.iterrows()]
    ax.set_yticklabels(labels)
    
    ax.set_xlabel('Δ Score (Найкраща конфігурація − Повний пайплайн)', fontsize=12)
    ax.set_title('Forest Plot: Ефект Over-engineering по датасетах (N=20)\nПозитивне значення = Спрощена конфігурація перевершує повну', fontsize=14)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Великий ефект (d ≥ 0.8)'),
        Patch(facecolor='#f39c12', label='Середній ефект (0.5 ≤ d < 0.8)'),
        Patch(facecolor='#3498db', label='Малий ефект (0.2 ≤ d < 0.5)'),
        Patch(facecolor='#95a5a6', label='Незначний (d < 0.2)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(-0.1, max(df_sorted['Delta']) + 0.1)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "forest_plot_all_datasets.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Збережено: {output_path}")
    
    return fig


def generate_markdown_table(df):
    """Generate markdown table with Cohen's d."""
    lines = [
        "| Датасет | Найкраща конфігурація | Score | Full Score | Δ | Cohen's d | Ефект |",
        "|:--------|:---------------------|------:|----------:|-----:|----------:|:------|"
    ]
    
    for _, row in df.sort_values('Cohen_d', ascending=False).iterrows():
        lines.append(
            f"| {row['Dataset']} | `{row['Best Config']}` | "
            f"{row['Best Score']:.3f} | {row['Full Score']:.3f} | "
            f"+{row['Delta']:.3f} | {row['Cohen_d']:.2f} | **{row['Effect Size']}** |"
        )
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Enhanced Analysis: Cohen's d & Forest Plot")
    print("=" * 60)
    
    # Load and process data
    df = load_and_process_data()
    
    print("\n📊 Effect Size Summary (N=20 runs):\n")
    print(df[['Dataset', 'Best Config', 'Delta', 'Cohen_d', 'Effect Size']].to_string(index=False))
    
    # Count by effect size
    print("\n📈 Effect Size Distribution:")
    print(df['Effect Size'].value_counts().to_string())
    
    # Create forest plot
    print("\n🎨 Creating Forest Plot...")
    create_forest_plot(df)
    
    # Generate markdown table
    print("\n📝 Markdown Table for Tezy:\n")
    table = generate_markdown_table(df)
    print(table)
    
    # Save table to file
    table_path = OUTPUT_DIR / "cohens_d_table.md"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("# Cohen's d Effect Size Analysis\n\n")
        f.write(f"Generated: 2025-11-25\n\n")
        f.write("## Summary Table (N=20 runs)\n\n")
        f.write(table)
        f.write("\n\n## Effect Size Classification\n")
        f.write("- **Large** (d ≥ 0.8): Practically significant, strong evidence\n")
        f.write("- **Medium** (0.5 ≤ d < 0.8): Moderate practical significance\n")
        f.write("- **Small** (0.2 ≤ d < 0.5): Detectable but limited practical significance\n")
        f.write("- **Negligible** (d < 0.2): Statistically significant but practically irrelevant\n")
    
    print(f"\nSaved: {table_path}")
    
    # Save CSV for reference
    csv_path = OUTPUT_DIR / "cohens_d_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
