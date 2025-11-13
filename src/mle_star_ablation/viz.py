"""
Модуль для візуалізації результатів абляційних експериментів.
"""

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
    save_path: Optional[str] = None
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
    
    plt.show()


def plot_boxplot(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = 'Accuracy',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
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
                order=ordered_configs, ax=ax, palette="husl")
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
    
    plt.show()


def plot_violin(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = 'Accuracy',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
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
                   order=ordered_configs, ax=ax, palette="husl")
    
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
    
    plt.show()


def plot_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = 'p_value',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Створює heatmap для попарних порівнянь.
    
    Args:
        comparison_df: DataFrame з результатами попарних порівнянь
        metric: Яку метрику візуалізувати ('p_value', 'cohen_d', тощо)
        figsize: Розмір фігури
        save_path: Шлях для збереження
    """
    # Створення квадратної матриці
    configs = sorted(set(comparison_df['config_a'].tolist() + 
                        comparison_df['config_b'].tolist()))
    
    matrix = pd.DataFrame(np.nan, index=configs, columns=configs)
    
    for _, row in comparison_df.iterrows():
        matrix.loc[row['config_a'], row['config_b']] = row[metric]
        matrix.loc[row['config_b'], row['config_a']] = row[metric]
    
    # Діагональ = 0 (порівняння з самим собою)
    for config in configs:
        matrix.loc[config, config] = 0 if metric == 'mean_diff' else 1.0
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric == 'p_value':
        # Логарифмічна шкала для p-values
        sns.heatmap(matrix.astype(float), annot=True, fmt='.3f', 
                    cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'P-value'})
    else:
        sns.heatmap(matrix.astype(float), annot=True, fmt='.3f',
                    cmap='coolwarm', center=0, ax=ax, 
                    cbar_kws={'label': metric})
    
    ax.set_title(f'Pairwise Comparison Heatmap ({metric})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_ablation_impact(
    results_df: pd.DataFrame,
    baseline_config: str,
    metric_name: str = 'accuracy',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
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
    
    plt.show()


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
        save_path=output_path / 'comparison_barplot.png'
    )
    
    # Boxplot
    plot_boxplot(
        results_dict, metric_name,
        save_path=output_path / 'comparison_boxplot.png'
    )
    
    # Violin plot
    plot_violin(
        results_dict, metric_name,
        save_path=output_path / 'comparison_violin.png'
    )
    
    # Heatmap (якщо є порівняння)
    if comparison_df is not None and len(comparison_df) > 0:
        plot_heatmap(
            comparison_df, metric='p_value',
            save_path=output_path / 'pvalue_heatmap.png'
        )
        
        if 'cohen_d' in comparison_df.columns:
            plot_heatmap(
                comparison_df, metric='cohen_d',
                save_path=output_path / 'cohend_heatmap.png'
            )
    
    print(f"All plots saved to {output_dir}/")
