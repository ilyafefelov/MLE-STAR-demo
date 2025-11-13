"""
MLE-STAR Ablation Analysis Framework

Фреймворк для абляційного аналізу та статистичної оцінки важливості 
компонентів ML-конвеєра, згенерованого офіційним Google ADK MLE-STAR.

АРХІТЕКТУРА:
1. Google ADK MLE-STAR генерує ML-конвеєр
2. Наш фреймворк виконує абляційний аналіз
3. Статистична оцінка значущості компонентів

Модулі:
- config: Конфігурації для абляційних експериментів (AblationConfig)
- mle_star_generated_pipeline: Адаптер для MLE-STAR згенерованого коду
- ablation_runner: Виконання експериментів та збір метрик
- datasets: Завантаження та обробка даних
- metrics: Обчислення метрик моделей
- stats: Статистичний аналіз результатів (t-test, ANOVA, Cohen's d)
- viz: Візуалізація результатів

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

__version__ = '0.2.0'
__author__ = 'Фефелов Ілля Олександрович'

# === ОСНОВНІ МОДУЛІ ===
from .config import (
    AblationConfig, 
    get_standard_configs, 
    get_cumulative_configs,
    create_custom_config,
)
from .mle_star_generated_pipeline import (
    build_full_pipeline,
    build_pipeline,
    inspect_pipeline,
    print_pipeline_structure,
)
from .ablation_runner import (
    run_single_config,
    run_ablation_suite,
    summarize_results,
    compare_to_baseline,
    save_results,
)

# === ДОПОМІЖНІ МОДУЛІ ===
from .datasets import DatasetLoader, list_available_datasets, load_dataset
from .metrics import calculate_classification_metrics, calculate_regression_metrics
from .stats import (
    paired_t_test, 
    anova_test, 
    generate_statistical_report, 
    pairwise_comparison, 
    summarize_statistics,
    bonferroni_correction,
    effect_size_cohen_d,
)
from .viz import plot_comparison_barplot, plot_boxplot, create_all_plots

__all__ = [
    # Конфігурації
    'AblationConfig',
    'get_standard_configs',
    'get_cumulative_configs',
    'create_custom_config',
    
    # MLE-STAR Generated Pipeline
    'build_full_pipeline',
    'build_pipeline',
    'inspect_pipeline',
    'print_pipeline_structure',
    
    # Абляційні експерименти
    'run_single_config',
    'run_ablation_suite',
    'summarize_results',
    'compare_to_baseline',
    'save_results',
    
    # Датасети
    'DatasetLoader',
    'load_dataset',
    'list_available_datasets',
    
    # Метрики
    'calculate_classification_metrics',
    'calculate_regression_metrics',
    
    # Статистика
    'paired_t_test',
    'anova_test',
    'generate_statistical_report',
    'pairwise_comparison',
    'summarize_statistics',
    'bonferroni_correction',
    'effect_size_cohen_d',
    
    # Візуалізація
    'plot_comparison_barplot',
    'plot_boxplot',
    'create_all_plots',
]
