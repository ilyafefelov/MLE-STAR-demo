"""
Модуль для статистичного аналізу результатів абляційних експериментів.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Виконує парний t-test для порівняння двох конфігурацій.
    
    Args:
        scores_a: Результати першої конфігурації (n повторів)
        scores_b: Результати другої конфігурації (n повторів)
        alpha: Рівень значущості
        
    Returns:
        Dict: Результати тесту (statistic, p-value, significant)
    """
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    
    return {
        't_statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'mean_diff': float(np.mean(scores_a) - np.mean(scores_b))
    }


def independent_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Виконує незалежний t-test для порівняння двох конфігурацій.
    
    Args:
        scores_a: Результати першої конфігурації
        scores_b: Результати другої конфігурації
        alpha: Рівень значущості
        
    Returns:
        Dict: Результати тесту
    """
    statistic, p_value = stats.ttest_ind(scores_a, scores_b)
    
    return {
        't_statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'mean_diff': float(np.mean(scores_a) - np.mean(scores_b))
    }


def anova_test(
    *score_groups: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Виконує ANOVA для порівняння кількох конфігурацій.
    
    Args:
        *score_groups: Групи результатів для кожної конфігурації
        alpha: Рівень значущості
        
    Returns:
        Dict: Результати тесту
    """
    statistic, p_value = stats.f_oneway(*score_groups)
    
    return {
        'f_statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'n_groups': len(score_groups)
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Застосовує поправку Бонферроні для множинних порівнянь.
    
    Args:
        p_values: Список p-values з різних тестів
        alpha: Оригінальний рівень значущості
        
    Returns:
        Dict: Скориговані результати
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_tests': n_tests,
        'significant_tests': [
            i for i, p in enumerate(p_values) if p < corrected_alpha
        ],
        'corrected_p_values': [p * n_tests for p in p_values]
    }


def effect_size_cohen_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> float:
    """
    Обчислює Cohen's d (розмір ефекту).
    
    Args:
        scores_a: Результати першої конфігурації
        scores_b: Результати другої конфігурації
        
    Returns:
        float: Cohen's d
    """
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)
    
    n_a = len(scores_a)
    n_b = len(scores_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    
    cohen_d = (mean_a - mean_b) / pooled_std if pooled_std != 0 else 0
    
    return float(cohen_d)


def confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Обчислює довірчий інтервал для середнього.
    
    Args:
        scores: Результати експериментів
        confidence: Рівень довіри (0.95 = 95%)
        
    Returns:
        Tuple[mean, lower_bound, upper_bound]
    """
    mean = np.mean(scores)
    sem = stats.sem(scores)  # Standard error of the mean
    interval = stats.t.interval(
        confidence,
        len(scores) - 1,
        loc=mean,
        scale=sem
    )
    
    return float(mean), float(interval[0]), float(interval[1])


def pairwise_comparison(
    results_dict: Dict[str, np.ndarray],
    alpha: float = 0.05,
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Виконує попарне порівняння всіх конфігурацій.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        alpha: Рівень значущості
        correction: Метод поправки ('bonferroni' або 'none')
        
    Returns:
        DataFrame: Таблиця з результатами порівнянь
    """
    config_names = list(results_dict.keys())
    n_configs = len(config_names)
    
    comparisons = []
    
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            config_a = config_names[i]
            config_b = config_names[j]
            
            scores_a = results_dict[config_a]
            scores_b = results_dict[config_b]
            
            # T-test
            t_result = paired_t_test(scores_a, scores_b, alpha)

            # Approximate z-statistic from two-tailed p-value to help compare tests on a common scale
            z_statistic = _t_to_z_score(t_result['t_statistic'], t_result['p_value'])
            
            # Effect size
            cohen_d = effect_size_cohen_d(scores_a, scores_b)
            
            comparisons.append({
                'config_a': config_a,
                'config_b': config_b,
                'mean_a': np.mean(scores_a),
                'mean_b': np.mean(scores_b),
                'mean_diff': t_result['mean_diff'],
                't_statistic': t_result['t_statistic'],
                'z_statistic': z_statistic,
                'p_value': t_result['p_value'],
                'cohen_d': cohen_d,
                'significant': t_result['significant']
            })
    
    df = pd.DataFrame(comparisons)
    
    # Застосувати поправку Бонферроні
    if correction == 'bonferroni' and len(df) > 0:
        corrected_alpha = alpha / len(df)
        df['bonferroni_significant'] = df['p_value'] < corrected_alpha
        df['corrected_alpha'] = corrected_alpha
    
    return df


def _t_to_z_score(t_statistic: float, p_value: float) -> float:
    """Convert a two-tailed t-test result to an equivalent z-score."""
    if np.isnan(t_statistic) or np.isnan(p_value):
        return float('nan')
    # Two-tailed p-value -> one-tailed probability for conversion
    clipped_p = np.clip(p_value / 2.0, 1e-12, 0.5)
    z_value = stats.norm.ppf(1 - clipped_p)
    if np.isnan(z_value) or np.isinf(z_value):
        return float('nan')
    return float(np.sign(t_statistic) * z_value)


def summarize_statistics(
    results_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Створює підсумкову статистику для всіх конфігурацій.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        
    Returns:
        DataFrame: Таблиця зі статистиками
    """
    summaries = []
    
    for config_name, scores in results_dict.items():
        mean, ci_lower, ci_upper = confidence_interval(scores)
        
        summaries.append({
            'configuration': config_name,
            'mean': mean,
            'std': np.std(scores, ddof=1),
            'min': np.min(scores),
            'max': np.max(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_runs': len(scores)
        })
    
    df = pd.DataFrame(summaries)
    df = df.sort_values('mean', ascending=False)
    
    return df


def interpret_cohen_d(cohen_d: float) -> str:
    """
    Інтерпретує розмір ефекту Cohen's d.
    
    Args:
        cohen_d: Значення Cohen's d
        
    Returns:
        str: Інтерпретація
    """
    abs_d = abs(cohen_d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_statistical_report(
    results_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> str:
    """
    Генерує повний статистичний звіт.
    
    Args:
        results_dict: Словник {config_name: scores_array}
        alpha: Рівень значущості
        
    Returns:
        str: Форматований звіт
    """
    report = "\n" + "="*70 + "\n"
    report += "STATISTICAL ANALYSIS REPORT\n"
    report += "="*70 + "\n\n"
    
    # Підсумкова статистика
    summary = summarize_statistics(results_dict)
    report += "SUMMARY STATISTICS:\n"
    report += "-"*70 + "\n"
    report += summary.to_string(index=False)
    report += "\n\n"
    
    # ANOVA
    if len(results_dict) > 2:
        anova_result = anova_test(*results_dict.values(), alpha=alpha)
        report += "ANOVA TEST:\n"
        report += "-"*70 + "\n"
        report += f"F-statistic: {anova_result['f_statistic']:.4f}\n"
        report += f"P-value: {anova_result['p_value']:.4e}\n"
        report += f"Significant (α={alpha}): {anova_result['significant']}\n"
        report += "\n"
    
    # Попарні порівняння
    if len(results_dict) >= 2:
        comparisons = pairwise_comparison(results_dict, alpha)
        report += "PAIRWISE COMPARISONS:\n"
        report += "-"*70 + "\n"
        report += comparisons.to_string(index=False)
        report += "\n\n"
    
    report += "="*70 + "\n"
    
    return report
