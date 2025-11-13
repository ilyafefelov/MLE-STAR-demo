"""
Модуль для обчислення метрик якості моделей.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix
)
from typing import Dict, Optional


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Обчислює метрики для класифікації.
    
    Args:
        y_true: Справжні мітки
        y_pred: Передбачені мітки
        y_proba: Ймовірності для кожного класу (для ROC-AUC)
        average: Метод усереднення для мультикласової класифікації
        
    Returns:
        Dict[str, float]: Словник з метриками
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    
    # ROC-AUC тільки якщо є ймовірності
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Бінарна класифікація
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # Мультикласова класифікація
                metrics['roc_auc'] = float(roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average=average
                ))
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = np.nan
    
    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Обчислює метрики для регресії.
    
    Args:
        y_true: Справжні значення
        y_pred: Передбачені значення
        
    Returns:
        Dict[str, float]: Словник з метриками
    """
    metrics = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2_score'] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    return metrics


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, any]:
    """
    Обчислює confusion matrix та пов'язані метрики.
    
    Args:
        y_true: Справжні мітки
        y_pred: Передбачені мітки
        
    Returns:
        Dict: Словник з confusion matrix та метриками
    """
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {}
    }
    
    # Метрики для кожного класу
    n_classes = cm.shape[0]
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        result['per_class_metrics'][f'class_{i}'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(tp + fn)
        }
    
    return result


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Форматує метрики в читабельний звіт.
    
    Args:
        metrics: Словник з метриками
        
    Returns:
        str: Форматований звіт
    """
    report = "\n" + "="*50 + "\n"
    report += "METRICS REPORT\n"
    report += "="*50 + "\n"
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            report += f"{metric_name.upper():20s}: {value:.4f}\n"
        else:
            report += f"{metric_name.upper():20s}: {value}\n"
    
    report += "="*50 + "\n"
    
    return report


def compare_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str = 'accuracy'
) -> str:
    """
    Порівнює метрики між різними конфігураціями.
    
    Args:
        metrics_dict: Словник {config_name: {metric: value}}
        metric_name: Назва метрики для порівняння
        
    Returns:
        str: Форматоване порівняння
    """
    report = f"\n{'='*60}\n"
    report += f"COMPARISON: {metric_name.upper()}\n"
    report += f"{'='*60}\n"
    
    # Сортуємо за значенням метрики
    sorted_configs = sorted(
        metrics_dict.items(),
        key=lambda x: x[1].get(metric_name, 0),
        reverse=True
    )
    
    for rank, (config_name, metrics) in enumerate(sorted_configs, 1):
        value = metrics.get(metric_name, np.nan)
        report += f"{rank}. {config_name:40s}: {value:.4f}\n"
    
    # Різниця між найкращим та найгіршим
    if len(sorted_configs) >= 2:
        best_value = sorted_configs[0][1].get(metric_name, 0)
        worst_value = sorted_configs[-1][1].get(metric_name, 0)
        diff = best_value - worst_value
        diff_pct = (diff / worst_value * 100) if worst_value != 0 else 0
        
        report += f"\n{'='*60}\n"
        report += f"Best vs Worst: {diff:.4f} ({diff_pct:.2f}% improvement)\n"
    
    report += f"{'='*60}\n"
    
    return report
