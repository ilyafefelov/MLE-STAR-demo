"""
Compute statistical tests for ablation study results.
Calculates t-test, p-values, Cohen's d with proper interpretation.
"""

import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path

# Dataset results from N=20 runs (extracted from aggregated_summary.csv)
# Format: {dataset: {config: (mean, std, n)}}
RESULTS = {
    # Classification datasets
    'cls_iris': {
        'full': (0.9083, 0.0494, 20),
        'no_fe': (0.9633, 0.0284, 20),
        'minimal': (0.9600, 0.0256, 20),
        'no_scaling': (0.9567, 0.0244, 20),
    },
    'cls_breast_cancer': {
        'full': (0.9487, 0.0174, 20),
        'no_fe': (0.9557, 0.0162, 20),
        'minimal': (0.9557, 0.0162, 20),
        'no_scaling': (0.8518, 0.0313, 20),
    },
    'cls_digits': {
        'full': (0.9790, 0.0078, 20),
        'no_fe': (0.9822, 0.0063, 20),
        'minimal': (0.8400, 0.0240, 20),
        'no_scaling': (0.9149, 0.0199, 20),
    },
    'cls_wine': {
        'full': (0.9722, 0.0255, 20),
        'no_fe': (0.9736, 0.0229, 20),
        'no_tuning': (0.9833, 0.0189, 20),
        'minimal': (0.6861, 0.0350, 20),
        'no_scaling': (0.7208, 0.0489, 20),
    },
    'cls_synthetic': {
        'full': (0.8835, 0.0234, 20),
        'no_fe': (0.8867, 0.0221, 20),
        'minimal': (0.8848, 0.0166, 20),
        'no_scaling': (0.8856, 0.0172, 20),
    },
    # Regression datasets
    'reg_california': {
        'full': (0.6366, 0.0116, 20),
        'no_fe': (0.7817, 0.0085, 20),
        'minimal': (0.7822, 0.0090, 20),
        'no_scaling': (-0.0007, 0.0015, 20),
    },
    'reg_diabetes': {
        'full': (0.4704, 0.0643, 20),
        'no_fe': (0.4594, 0.0659, 20),
        'no_scaling': (0.4720, 0.0620, 20),
        'minimal': (0.4608, 0.0675, 20),
    },
    'reg_synth_easy': {
        'full': (0.9575, 0.0148, 20),
        'no_fe': (0.9810, 0.0022, 20),
        'minimal': (0.9814, 0.0020, 20),
        'no_scaling': (0.9559, 0.0172, 20),
    },
    'reg_synth_medium': {
        'full': (0.9050, 0.0343, 20),
        'no_fe': (0.9470, 0.0044, 20),
        'minimal': (0.9475, 0.0031, 20),
        'no_scaling': (0.9124, 0.0349, 20),
    },
    'reg_synth_nonlinear': {
        'full': (0.5030, 0.0539, 20),
        'no_fe': (0.8141, 0.0247, 20),
        'minimal': (0.8141, 0.0240, 20),
        'no_scaling': (0.4954, 0.0560, 20),
    },
}


def cohens_d(mean1, std1, n1, mean2, std2, n2):
    """Calculate Cohen's d effect size."""
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def cohens_d_ci(d, n1, n2, alpha=0.05):
    """Calculate 95% CI for Cohen's d using non-central t approximation."""
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    t_crit = stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)
    return (d - t_crit * se_d, d + t_crit * se_d)


def welch_ttest(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test for unequal variances."""
    # t-statistic
    se1 = std1**2 / n1
    se2 = std2**2 / n2
    se_diff = np.sqrt(se1 + se2)
    if se_diff == 0:
        return 0, 1.0
    t_stat = (mean1 - mean2) / se_diff
    
    # Degrees of freedom (Welch-Satterthwaite)
    df = (se1 + se2)**2 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return t_stat, p_value


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_p_value(p):
    """Interpret p-value significance."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def find_best_config(dataset_results):
    """Find the best performing configuration."""
    best_config = max(dataset_results.items(), key=lambda x: x[1][0])
    return best_config[0], best_config[1]


def analyze_dataset(name, results):
    """Analyze a single dataset."""
    full_data = results.get('full')
    if not full_data:
        return None
    
    best_config, best_data = find_best_config(results)
    
    # Skip if full is the best
    if best_config == 'full':
        best_config = 'no_fe' if 'no_fe' in results else 'minimal'
        best_data = results[best_config]
    
    # Calculate statistics
    t_stat, p_value = welch_ttest(
        best_data[0], best_data[1], best_data[2],
        full_data[0], full_data[1], full_data[2]
    )
    
    d = cohens_d(
        best_data[0], best_data[1], best_data[2],
        full_data[0], full_data[1], full_data[2]
    )
    
    d_ci = cohens_d_ci(d, best_data[2], full_data[2])
    
    delta = best_data[0] - full_data[0]
    
    return {
        'dataset': name,
        'best_config': best_config,
        'best_score': best_data[0],
        'best_std': best_data[1],
        'full_score': full_data[0],
        'full_std': full_data[1],
        'delta': delta,
        't_statistic': t_stat,
        'p_value': p_value,
        'p_interpret': interpret_p_value(p_value),
        'cohens_d': d,
        'd_ci_lower': d_ci[0],
        'd_ci_upper': d_ci[1],
        'd_interpret': interpret_cohens_d(d),
        'n': best_data[2],
    }


def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS OF ABLATION STUDY RESULTS")
    print("=" * 80)
    print()
    
    results_list = []
    
    for dataset, configs in RESULTS.items():
        analysis = analyze_dataset(dataset, configs)
        if analysis:
            results_list.append(analysis)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Print summary table
    print("TABLE: Statistical comparison of simplified vs full configurations (N=20)")
    print("-" * 120)
    print(f"{'Dataset':<20} {'Best Config':<12} {'Best±Std':<14} {'Full±Std':<14} {'Δ':<8} {'t':<8} {'p-value':<12} {'d':<8} {'95% CI d':<16} {'Effect':<10}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['dataset']:<20} "
              f"{row['best_config']:<12} "
              f"{row['best_score']:.3f}±{row['best_std']:.3f}  "
              f"{row['full_score']:.3f}±{row['full_std']:.3f}  "
              f"{row['delta']:+.3f}  "
              f"{row['t_statistic']:>6.2f}  "
              f"{row['p_value']:<.2e}{row['p_interpret']:<3} "
              f"{row['cohens_d']:>6.2f}  "
              f"[{row['d_ci_lower']:.2f}, {row['d_ci_upper']:.2f}]  "
              f"{row['d_interpret']:<10}")
    
    print("-" * 120)
    print()
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"  Total datasets analyzed: {len(df)}")
    print(f"  Datasets with Large effect (d ≥ 0.8): {len(df[df['d_interpret'] == 'Large'])}")
    print(f"  Datasets with Medium effect (0.5 ≤ d < 0.8): {len(df[df['d_interpret'] == 'Medium'])}")
    print(f"  Datasets with Small effect (0.2 ≤ d < 0.5): {len(df[df['d_interpret'] == 'Small'])}")
    print(f"  Datasets with Negligible effect (d < 0.2): {len(df[df['d_interpret'] == 'Negligible'])}")
    print()
    print(f"  Statistically significant (p < 0.05): {len(df[df['p_value'] < 0.05])}/{len(df)}")
    print(f"  Highly significant (p < 0.01): {len(df[df['p_value'] < 0.01])}/{len(df)}")
    print(f"  Very highly significant (p < 0.001): {len(df[df['p_value'] < 0.001])}/{len(df)}")
    print()
    
    # Average Cohen's d
    print(f"  Mean Cohen's d: {df['cohens_d'].mean():.2f}")
    print(f"  Median Cohen's d: {df['cohens_d'].median():.2f}")
    print(f"  Mean Δ (score improvement): {df['delta'].mean():.3f}")
    print()
    
    # Generate markdown table for thesis
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE FOR THESIS")
    print("=" * 80)
    print()
    print("**Таблиця 2. Статистичний аналіз переваги спрощених конфігурацій (N=20)**")
    print()
    print("| Датасет | Найкраща | Score±Std | Full±Std | Δ | t | p-value | d | 95% CI | Ефект |")
    print("|---------|----------|-----------|----------|---|---|---------|---|--------|-------|")
    
    for _, row in df.iterrows():
        dataset_short = row['dataset'].replace('cls_', '').replace('reg_', '')
        print(f"| {dataset_short} | `{row['best_config']}` | "
              f"{row['best_score']:.3f}±{row['best_std']:.3f} | "
              f"{row['full_score']:.3f}±{row['full_std']:.3f} | "
              f"{row['delta']:+.3f} | "
              f"{row['t_statistic']:.2f} | "
              f"{row['p_value']:.2e}{row['p_interpret']} | "
              f"{row['cohens_d']:.2f} | "
              f"[{row['d_ci_lower']:.2f}, {row['d_ci_upper']:.2f}] | "
              f"**{row['d_interpret']}** |")
    
    print()
    print("*Значущість: *** p<0.001, ** p<0.01, * p<0.05, ns - незначущий*")
    print("*Класифікація Cohen's d: Large (d≥0.8), Medium (0.5≤d<0.8), Small (0.2≤d<0.5), Negligible (d<0.2)*")
    print()
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / 'results' / 'statistical_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    main()
