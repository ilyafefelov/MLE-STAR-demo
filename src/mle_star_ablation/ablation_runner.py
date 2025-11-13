# src/mle_star_ablation/ablation_runner.py

"""
Модуль для виконання абляційних експериментів на MLE-STAR pipeline.

Цей модуль відповідає за:
- Запуск експериментів з різними конфігураціями
- Багаторазове тестування для статистичної значущості
- Збір метрик (accuracy, F1, ROC-AUC, час тренування)
- Збереження результатів у структурованому форматі

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import time
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold, 
    cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

from .config import AblationConfig, get_standard_configs
from .mle_star_generated_pipeline import build_pipeline


# ==== 1. ЗАПУСК ОДНОГО ЕКСПЕРИМЕНТУ ========================================

def run_single_config(
    X: np.ndarray,
    y: np.ndarray,
    config: AblationConfig,
    n_folds: int = 5,
    random_state: int = 42,
    task_type: str = "classification",
    scoring: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Запускає один експеримент з заданою конфігурацією.
    
    Використовує K-fold cross-validation для оцінки якості pipeline.
    Рахує множину метрик та час виконання.
    
    Args:
        X: Матриця ознак (n_samples, n_features)
        y: Цільова змінна (n_samples,)
        config: Конфігурація абляції (які компоненти увімкнені)
        n_folds: Кількість фолдів для cross-validation
        random_state: Seed для відтворюваності
        task_type: Тип задачі ("classification" або "regression")
        scoring: Список метрик для оцінки (None = дефолтні)
        
    Returns:
        dict: Результати експерименту з метриками
        
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> config = AblationConfig(name="full")
        >>> results = run_single_config(X, y, config, n_folds=3)
        >>> print(f"Accuracy: {results['mean_accuracy']:.3f}")
    """
    # Підготовка cross-validation
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'r2']
    
    # Побудова pipeline
    try:
        pipeline = build_pipeline(config)
    except Exception as e:
        warnings.warn(f"Помилка побудови pipeline для {config.name}: {e}")
        return {
            'config_name': config.name,
            'error': str(e),
            'status': 'failed',
            **config.to_dict()
        }
    
    # Cross-validation з метриками
    start_time = time.time()
    
    try:
        cv_results = cross_validate(
            pipeline, 
            X, 
            y, 
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,  # Паралелізація
            error_score='raise'
        )
        elapsed_time = time.time() - start_time
        
    except Exception as e:
        warnings.warn(f"Помилка під час cross-validation для {config.name}: {e}")
        return {
            'config_name': config.name,
            'error': str(e),
            'status': 'failed',
            'time_sec': time.time() - start_time,
            **config.to_dict()
        }
    
    # Агрегація результатів
    results = {
        'config_name': config.name,
        'status': 'success',
        'n_folds': n_folds,
        'time_sec': elapsed_time,
        'time_per_fold_sec': elapsed_time / n_folds,
    }
    
    # Додаємо параметри конфігурації
    results.update(config.to_dict())
    
    # Додаємо метрики
    for metric_name, scores in cv_results.items():
        if metric_name.startswith('test_'):
            clean_name = metric_name.replace('test_', '')
            results[f'mean_{clean_name}'] = float(np.mean(scores))
            results[f'std_{clean_name}'] = float(np.std(scores))
            results[f'min_{clean_name}'] = float(np.min(scores))
            results[f'max_{clean_name}'] = float(np.max(scores))
    
    return results


# ==== 2. ЗАПУСК СЕРІЇ ЕКСПЕРИМЕНТІВ ========================================

def run_ablation_suite(
    X: np.ndarray,
    y: np.ndarray,
    configs: Optional[List[AblationConfig]] = None,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    task_type: str = "classification",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Запускає повний набір абляційних експериментів.
    
    Для кожної конфігурації:
    1. Виконує n_repeats прогонів з різними random seeds
    2. Збирає метрики для кожного прогону
    3. Агрегує результати у DataFrame
    
    Args:
        X: Матриця ознак
        y: Цільова змінна
        configs: Список конфігурацій (None = стандартні)
        n_folds: Кількість фолдів для CV
        n_repeats: Кількість повторів кожної конфігурації
        random_state: Базовий seed
        task_type: Тип задачі
        verbose: Чи виводити прогрес
        
    Returns:
        pd.DataFrame: Результати всіх експериментів
        
    Example:
        >>> configs = get_standard_configs()
        >>> results = run_ablation_suite(X, y, configs, n_repeats=3)
        >>> print(results.groupby('config_name')['mean_accuracy'].mean())
    """
    if configs is None:
        configs = get_standard_configs()
    
    all_results = []
    total_experiments = len(configs) * n_repeats
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ЗАПУСК АБЛЯЦІЙНОГО АНАЛІЗУ")
        print(f"{'='*70}")
        print(f"Конфігурацій: {len(configs)}")
        print(f"Повторів: {n_repeats}")
        print(f"Фолдів: {n_folds}")
        print(f"Всього експериментів: {total_experiments}")
        print(f"{'='*70}\n")
    
    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"[{i}/{len(configs)}] Конфігурація: {config.name}")
            print(f"  {config.description}")
        
        for repeat in range(n_repeats):
            # Різний seed для кожного повтору
            seed = random_state + repeat
            
            if verbose and n_repeats > 1:
                print(f"  Повтор {repeat + 1}/{n_repeats} (seed={seed})...", end=" ")
            
            result = run_single_config(
                X, y, config,
                n_folds=n_folds,
                random_state=seed,
                task_type=task_type
            )
            
            # Додаємо інформацію про повтор
            result['repeat'] = repeat
            result['random_seed'] = seed
            
            all_results.append(result)
            
            if verbose and n_repeats > 1:
                status = result.get('status', 'unknown')
                if status == 'success':
                    # Виводимо основну метрику
                    main_metric = 'mean_accuracy' if task_type == 'classification' else 'mean_r2'
                    if main_metric in result:
                        print(f"✓ {main_metric.replace('mean_', '')}: {result[main_metric]:.4f}")
                    else:
                        print("✓")
                else:
                    print(f"✗ {result.get('error', 'unknown error')}")
        
        if verbose:
            print()
    
    df = pd.DataFrame(all_results)
    
    if verbose:
        print(f"{'='*70}")
        print(f"ЗАВЕРШЕНО: {len(df)} експериментів")
        successful = len(df[df['status'] == 'success'])
        print(f"Успішних: {successful}/{len(df)}")
        print(f"{'='*70}\n")
    
    return df


# ==== 3. АНАЛІЗ РЕЗУЛЬТАТІВ ================================================

def summarize_results(
    results_df: pd.DataFrame,
    group_by: str = 'config_name',
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Агрегує результати експериментів по групах.
    
    Args:
        results_df: DataFrame з результатами run_ablation_suite()
        group_by: Поле для групування (зазвичай 'config_name')
        metrics: Список метрик для агрегації (None = всі mean_*)
        
    Returns:
        pd.DataFrame: Агреговані статистики
        
    Example:
        >>> summary = summarize_results(results_df)
        >>> print(summary[['config_name', 'accuracy_mean', 'accuracy_std']])
    """
    # Фільтруємо тільки успішні експерименти
    df = results_df[results_df['status'] == 'success'].copy()
    
    if len(df) == 0:
        warnings.warn("Немає успішних експериментів для аналізу")
        return pd.DataFrame()
    
    # Визначаємо метрики
    if metrics is None:
        metrics = [col for col in df.columns if col.startswith('mean_')]
    
    # Групування та агрегація
    agg_dict = {}
    for metric in metrics:
        if metric in df.columns:
            clean_name = metric.replace('mean_', '')
            agg_dict[f'{clean_name}_mean'] = (metric, 'mean')
            agg_dict[f'{clean_name}_std'] = (metric, 'std')
            agg_dict[f'{clean_name}_min'] = (metric, 'min')
            agg_dict[f'{clean_name}_max'] = (metric, 'max')
    
    # Додаємо час виконання
    if 'time_sec' in df.columns:
        agg_dict['time_mean'] = ('time_sec', 'mean')
        agg_dict['time_std'] = ('time_sec', 'std')
    
    summary = df.groupby(group_by).agg(**agg_dict).reset_index()
    
    # Додаємо кількість експериментів
    summary['n_experiments'] = df.groupby(group_by).size().values
    
    return summary


def compare_to_baseline(
    results_df: pd.DataFrame,
    baseline_name: str = 'full',
    metric: str = 'mean_accuracy'
) -> pd.DataFrame:
    """
    Порівнює всі конфігурації з baseline.
    
    Args:
        results_df: DataFrame з результатами
        baseline_name: Назва baseline конфігурації
        metric: Метрика для порівняння
        
    Returns:
        pd.DataFrame: Результати з відносними змінами
        
    Example:
        >>> comparison = compare_to_baseline(results_df, baseline_name='full')
        >>> print(comparison[['config_name', 'delta_accuracy', 'delta_pct']])
    """
    df = results_df[results_df['status'] == 'success'].copy()
    
    # Отримуємо baseline значення
    baseline_df = df[df['config_name'] == baseline_name]
    if len(baseline_df) == 0:
        raise ValueError(f"Baseline конфігурація '{baseline_name}' не знайдена")
    
    baseline_value = baseline_df[metric].mean()
    
    # Рахуємо відхилення
    summary = df.groupby('config_name')[metric].agg(['mean', 'std', 'count']).reset_index()
    summary['baseline_value'] = baseline_value
    summary[f'delta_{metric}'] = summary['mean'] - baseline_value
    summary['delta_pct'] = (summary[f'delta_{metric}'] / baseline_value) * 100
    
    # Сортуємо за впливом
    summary = summary.sort_values('delta_pct', ascending=False)
    
    return summary


# ==== 4. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ ============================================

def save_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = 'ablation'
) -> Dict[str, Path]:
    """
    Зберігає результати у різних форматах.
    
    Args:
        results_df: DataFrame з результатами
        output_dir: Директорія для збереження
        prefix: Префікс для файлів
        
    Returns:
        dict: Шляхи до збережених файлів
        
    Example:
        >>> paths = save_results(results_df, Path('results'))
        >>> print(paths['csv'])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # 1. Повні результати (CSV)
    csv_path = output_dir / f'{prefix}_full_results.csv'
    results_df.to_csv(csv_path, index=False)
    paths['csv'] = csv_path
    
    # 2. Агреговані результати
    summary = summarize_results(results_df)
    summary_path = output_dir / f'{prefix}_summary.csv'
    summary.to_csv(summary_path, index=False)
    paths['summary'] = summary_path
    
    # 3. Порівняння з baseline (якщо є)
    if 'full' in results_df['config_name'].values:
        comparison = compare_to_baseline(results_df)
        comparison_path = output_dir / f'{prefix}_comparison.csv'
        comparison.to_csv(comparison_path, index=False)
        paths['comparison'] = comparison_path
    
    return paths


# ==== ЕКСПОРТ ==============================================================

__all__ = [
    'run_single_config',
    'run_ablation_suite',
    'summarize_results',
    'compare_to_baseline',
    'save_results',
]
