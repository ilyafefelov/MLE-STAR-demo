"""
Модуль для візуалізації результатів абляційних експериментів.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


# Налаштування стилю
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_comparison_barplot(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = 'Accuracy',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Створює барчарт з довірчими інтервалами для порівняння конфігурацій.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        metric_name: Назва метрики для осі Y
        figsize: Розмір фігури
        save_path: Шлях для збереження (якщо None - тільки показати)
    """
    # Підготовка даних
    configs = list(results_dict.keys())
    means = [np.mean(scores) for scores in results_dict.values()]
    stds = [np.std(scores, ddof=1) for scores in results_dict.values()]
    
    # Сортування за середнім значенням
    sorted_indices = np.argsort(means)[::-1]
    configs = [configs[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=sns.color_palette("husl", len(configs)))
    
    # Додавання значень на стовпчиках
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison Across Configurations', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    # Close figure to avoid resource leak and blocking
    plt.close(fig)


def plot_boxplot(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = 'Accuracy',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
    , show: bool = False
):
    """
    Створює boxplot для порівняння розподілів.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        metric_name: Назва метрики
        figsize: Розмір фігури
        save_path: Шлях для збереження
    """
    # Підготовка даних для boxplot
    data = []
    labels = []
    
    for config_name, scores in results_dict.items():
        data.extend(scores)
        labels.extend([config_name] * len(scores))
    
    df = pd.DataFrame({'Configuration': labels, metric_name: data})
    
    # Сортування за медіаною
    medians = df.groupby('Configuration')[metric_name].median().sort_values(ascending=False)
    ordered_configs = medians.index.tolist()
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df, x='Configuration', y=metric_name, 
                order=ordered_configs, ax=ax)
    sns.stripplot(data=df, x='Configuration', y=metric_name,
                  order=ordered_configs, ax=ax, 
                  color='black', alpha=0.3, size=4)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Distribution Across Configurations', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_violin(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = 'Accuracy',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
    , show: bool = False
):
    """
    Створює violin plot для детального аналізу розподілів.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        metric_name: Назва метрики
        figsize: Розмір фігури
        save_path: Шлях для збереження
    """
    # Підготовка даних
    data = []
    labels = []
    
    for config_name, scores in results_dict.items():
        data.extend(scores)
        labels.extend([config_name] * len(scores))
    
    df = pd.DataFrame({'Configuration': labels, metric_name: data})
    
    # Сортування за медіаною
    medians = df.groupby('Configuration')[metric_name].median().sort_values(ascending=False)
    ordered_configs = medians.index.tolist()
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.violinplot(data=df, x='Configuration', y=metric_name,
                   order=ordered_configs, ax=ax)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Distribution (Violin Plot)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _build_pairwise_matrix(
    comparison_df: pd.DataFrame,
    metric: str
) -> pd.DataFrame:
    configs = sorted(set(comparison_df['config_a'].tolist() + comparison_df['config_b'].tolist()))
    matrix = pd.DataFrame(np.nan, index=configs, columns=configs, dtype=float)
    for _, row in comparison_df.iterrows():
        value = row.get(metric)
        if pd.isna(value):
            continue
        matrix.loc[row['config_a'], row['config_b']] = value
        matrix.loc[row['config_b'], row['config_a']] = value
    diag_value = 1.0 if metric == 'p_value' else 0.0
    np.fill_diagonal(matrix.values, diag_value)
    if metric != 'p_value':
        matrix = matrix.fillna(0.0)
    return matrix


def plot_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = 'p_value',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
    , show: bool = False
):
    """Створює heatmap для попарних порівнянь."""
    if comparison_df is None or comparison_df.empty or metric not in comparison_df.columns:
        return

    matrix = _build_pairwise_matrix(comparison_df, metric)

    fig, ax = plt.subplots(figsize=figsize)
    if metric == 'p_value':
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'P-value'})
    else:
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'label': metric.replace('_', ' ').title()})

    ax.set_title(f'Pairwise Comparison Heatmap ({metric.replace("_", " ").title()})',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_statistical_overview(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
    show: bool = False
):
    """Створює зведений макет heatmap для кількох статистик (t, p, z, d, mean diff)."""
    if comparison_df is None or comparison_df.empty:
        return
    default_metrics = ['p_value', 't_statistic', 'z_statistic', 'cohen_d', 'mean_diff']
    metrics = metrics or default_metrics
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, metric in enumerate(available_metrics):
        matrix = _build_pairwise_matrix(comparison_df, metric)
        ax = axes[idx]
        if metric == 'p_value':
            heatmap = sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                                  cbar_kws={'label': 'P-value'})
        elif metric in {'t_statistic', 'z_statistic', 'mean_diff', 'cohen_d'}:
            heatmap = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                                  cbar_kws={'label': metric.replace('_', ' ').title()})
        else:
            heatmap = sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                                  cbar_kws={'label': metric})
        heatmap.set_title(f'{metric.replace("_", " ").title()} Heatmap', fontsize=12, fontweight='bold')
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    for ax in axes[len(available_metrics):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_ablation_impact(
    results_df: pd.DataFrame,
    baseline_config: str,
    metric_name: str = 'accuracy',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
    , show: bool = False
):
    """
    Візуалізує вплив додавання кожного компонента відносно baseline.
    
    Args:
        results_df: DataFrame з результатами (колонки: config, metric_name)
        baseline_config: Назва базової конфігурації
        metric_name: Назва метрики
        figsize: Розмір фігури
        save_path: Шлях для збереження
    """
    baseline_score = results_df[results_df['configuration'] == baseline_config][metric_name].values[0]
    
    # Обчислення різниці
    results_df = results_df.copy()
    results_df['improvement'] = results_df[metric_name] - baseline_score
    results_df['improvement_pct'] = (results_df['improvement'] / baseline_score) * 100
    
    # Видалити baseline з графіку
    plot_df = results_df[results_df['configuration'] != baseline_config].copy()
    plot_df = plot_df.sort_values('improvement', ascending=True)
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if x > 0 else 'red' for x in plot_df['improvement']]
    bars = ax.barh(range(len(plot_df)), plot_df['improvement'], color=colors, alpha=0.7)
    
    # Додавання значень
    for i, (bar, imp, pct) in enumerate(zip(bars, plot_df['improvement'], plot_df['improvement_pct'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{imp:+.3f} ({pct:+.1f}%)',
                ha='left' if width > 0 else 'right',
                va='center', fontsize=9)
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['configuration'])
    ax.set_xlabel(f'Improvement over {baseline_config}', fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Impact Analysis ({metric_name})', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def create_all_plots(
    results_dict: Dict[str, np.ndarray],
    comparison_df: Optional[pd.DataFrame] = None,
    output_dir: str = 'results',
    metric_name: str = 'Accuracy'
):
    """
    Створює всі графіки та зберігає їх.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        comparison_df: DataFrame з попарними порівняннями
        output_dir: Директорія для збереження
        metric_name: Назва метрики
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating plots in {output_dir}/...")
    
    # Барчарт
    plot_comparison_barplot(
        results_dict, metric_name,
        save_path=output_path / 'comparison_barplot.png',
        show=False
    )
    
    # Boxplot
    plot_boxplot(
        results_dict, metric_name,
        save_path=output_path / 'comparison_boxplot.png',
        show=False
    )
    
    # Violin plot
    plot_violin(
        results_dict, metric_name,
        save_path=output_path / 'comparison_violin.png',
        show=False
    )
    
    # Heatmap (якщо є порівняння)
    if comparison_df is not None and len(comparison_df) > 0:
        heatmap_specs = [
            ('p_value', 'pvalue_heatmap.png'),
            ('t_statistic', 'tstat_heatmap.png'),
            ('z_statistic', 'zscore_heatmap.png'),
            ('cohen_d', 'cohend_heatmap.png'),
            ('mean_diff', 'mean_diff_heatmap.png')
        ]
        for metric, filename in heatmap_specs:
            if metric in comparison_df.columns:
                plot_heatmap(
                    comparison_df,
                    metric=metric,
                    save_path=output_path / filename,
                    show=False
                )

        plot_statistical_overview(
            comparison_df,
            save_path=output_path / 'statistical_overview.png',
            show=False
        )
    
    print(f"All plots saved to {output_dir}/")
