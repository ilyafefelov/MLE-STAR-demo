#!/usr/bin/env python
"""
Основний скрипт для запуску абляційних експериментів.
"""

import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Додаємо src до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mle_star_ablation.config import get_standard_configs
from src.mle_star_ablation.mle_star_generated_pipeline import build_pipeline
from src.mle_star_ablation.datasets import DatasetLoader
from src.mle_star_ablation.metrics import calculate_classification_metrics, calculate_regression_metrics
from src.mle_star_ablation.stats import (
    generate_statistical_report,
    pairwise_comparison,
    summarize_statistics
)
from src.mle_star_ablation.viz import create_all_plots
from src.mle_star_ablation.ast_utils import detect_data_leakage
from src.mle_star_ablation.ablation_runner import run_baseline_config


def run_single_experiment(config, X_train, X_test, y_train, y_test, random_state=42, deterministic=False, is_regression=False):
    """
    Запускає один експеримент з заданою конфігурацією.
    
    Args:
        config: AblationConfig
        X_train, X_test, y_train, y_test: Дані
        random_state: Seed (не використовується, бо random_state вже в Gemini pipeline)
        
    Returns:
        dict: Результати експерименту
    """
    pipeline = build_pipeline(config, random_state=random_state, deterministic=deterministic)
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Передбачення
    y_pred = pipeline.predict(X_test)
    
    # Ймовірності (якщо доступні)
    try:
        y_proba = pipeline.predict_proba(X_test)
    except:
        y_proba = None
    
    # Обчислення метрик
    if is_regression:
        metrics = calculate_regression_metrics(y_test, y_pred)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
    metrics['train_time'] = train_time
    
    return metrics


def run_multiple_runs(config, dataset_name, csv_path, target_column, n_runs=5, base_seed: int = None, deterministic: bool = False, forced_task_type: str = None):
    """
    Запускає кілька повторів експерименту з різними seed.
    
    Args:
        config: AblationConfig
        dataset_name: Назва датасету
        csv_path: Шлях до CSV
        target_column: Цільова колонка
        n_runs: Кількість повторів
        
    Returns:
        dict: Результати всіх повторів
    """
    all_results = []
    
    for run_idx in range(n_runs):
        if base_seed is not None:
            random_state = base_seed + run_idx
        else:
            random_state = 42 + run_idx
        
        # Завантажуємо дані з новим seed
        X_train, X_test, y_train, y_test = DatasetLoader.load_dataset(
            dataset_name=dataset_name,
            csv_path=csv_path,
            target_column=target_column,
            random_state=random_state
        )
        
        # Detect regression vs classification (unless forced)
        is_regression = False
        try:
            # if target is continuous type (float) or many unique values, treat as regression
            if forced_task_type is not None:
                is_regression = True if forced_task_type == 'regression' else False
            else:
                if np.issubdtype(y_train.dtype, np.floating) or len(np.unique(y_train)) > 20:
                    is_regression = True
        except Exception:
            is_regression = False

        # Check if config is 'baseline' (special case)
        if config == 'baseline':
             # Use run_baseline_config logic but adapted for train/test split (run_baseline_config uses CV)
             # Actually, run_baseline_config uses CV on X, y. Here we have X_train, X_test.
             # We should implement a simple baseline fit/predict here.
             from sklearn.dummy import DummyClassifier, DummyRegressor
             
             if is_regression:
                 dummy = DummyRegressor(strategy='mean')
             else:
                 dummy = DummyClassifier(strategy='most_frequent', random_state=random_state)
             
             start_time = time.time()
             dummy.fit(X_train, y_train)
             train_time = time.time() - start_time
             y_pred = dummy.predict(X_test)
             y_proba = None
             if not is_regression:
                 try:
                     y_proba = dummy.predict_proba(X_test)
                 except:
                     pass
             
             if is_regression:
                 metrics = calculate_regression_metrics(y_test, y_pred)
             else:
                 metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
             metrics['train_time'] = train_time
             metrics['config_name'] = 'baseline'
             metrics['description'] = 'Dummy Baseline'
             
        else:
            # Запускаємо експеримент
            metrics = run_single_experiment(
                config, X_train, X_test, y_train, y_test, random_state, deterministic=deterministic, is_regression=is_regression
            )
        
        metrics['run'] = run_idx + 1
        metrics['random_state'] = random_state
        all_results.append(metrics)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Запуск абляційного аналізу ML-конвеєра',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Параметри датасету
    parser.add_argument(
        '--dataset',
        type=str,
        default='breast_cancer',
        help='Назва датасету (breast_cancer, wine, digits, iris) або custom'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Шлях до CSV файлу (для dataset=custom)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Назва цільової колонки (для CSV)'
    )
    
    # Параметри експерименту
    parser.add_argument(
        '--n-runs',
        type=int,
        default=5,
        help='Кількість повторів кожної конфігурації'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Шлях до YAML конфігурації (опціонально)'
    )
    
    # Параметри виводу
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Директорія для збереження результатів'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Не створювати графіки'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Детальний вивід'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional base seed for deterministic runs'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Attempt to enforce deterministic runs (set threads to 1, seed RNGs)'
    )
    parser.add_argument(
        '--pipeline-file',
        type=str,
        default=None,
        help='Optional path to a Python file with build_full_pipeline() to use (overrides module pipeline)'
    )
    parser.add_argument(
        '--task-type',
        type=str,
        choices=['classification', 'regression'],
        default=None,
        help='Force the task type to classification or regression (overrides automatic detection)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['flash_lite', 'flash', 'pro', 'all'],
        default=None,
        help='Optional pre-defined variant name to use from model_comparison_results'
    )
    
    args = parser.parse_args()
    
    # Set deterministic mode if requested
    if args.seed is not None:
        import random
        # import numpy as np  <-- Removed to avoid UnboundLocalError
        random.seed(args.seed)
        np.random.seed(args.seed)
    if args.deterministic:
        # Try to force single-threaded BLAS to reduce non-determinism
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    # Dynamic pipeline file registration logic
    pipeline_file = args.pipeline_file
    if args.variant and not pipeline_file:
        # Map variant to file name pattern; expect files in model_comparison_results
        variant_map = {
            'flash_lite': f'model_comparison_results/gemini_2.5_flash_lite_{args.dataset}.py',
            'flash': f'model_comparison_results/gemini_2.5_flash_{args.dataset}.py',
            'pro': f'model_comparison_results/gemini_2.5_pro_{args.dataset}.py'
        }
        pipeline_file = variant_map.get(args.variant)

    if pipeline_file:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('custom_pipeline', pipeline_file)
            custom_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_mod)

            # Register custom build_full_pipeline in mle_star_generated_pipeline
            from src.mle_star_ablation import mle_star_generated_pipeline as mgp
            if hasattr(custom_mod, 'build_full_pipeline'):
                mgp.set_full_pipeline_callable(custom_mod.build_full_pipeline)
                print(f"Registered external pipeline: {pipeline_file}")
                
                # Check for data leakage
                try:
                    with open(pipeline_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    if detect_data_leakage(source_code):
                        print("\n" + "!"*70)
                        print("WARNING: POTENTIAL DATA LEAKAGE DETECTED!")
                        print(f"The pipeline file '{pipeline_file}' contains calls to .fit() or .fit_transform()")
                        print("outside of the pipeline definition. This may invalidate results.")
                        print("!"*70 + "\n")
                except Exception as e:
                    print(f"Warning: Could not check for data leakage: {e}")
            else:
                print(f"Warning: pipeline file {pipeline_file} does not define build_full_pipeline().")
        except Exception as e:
            print(f"Could not load pipeline file {pipeline_file}: {e}")

    # Створення директорії для результатів
    output_dir = Path(args.output_dir)
    # add variant suffix for output folder
    if args.variant:
        output_dir = output_dir / args.variant
    elif args.pipeline_file:
        output_dir = output_dir / Path(args.pipeline_file).stem
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Виведення інформації про датасет
    print("="*70)
    print("ABLATION ANALYSIS: MLE-STAR ML PIPELINE")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    if args.csv_path:
        print(f"CSV Path: {args.csv_path}")
        print(f"Target Column: {args.target}")
    print(f"Number of runs per config: {args.n_runs}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Інформація про датасет
    try:
        dataset_info = DatasetLoader.get_dataset_info(
            args.dataset, args.csv_path, args.target
        )
        print(f"Dataset info:")
        print(f"  - Samples: {dataset_info['n_samples']}")
        print(f"  - Features: {dataset_info['n_features']}")
        print(f"  - Classes: {dataset_info['n_classes']}")
        print(f"  - Train/Test: {dataset_info['train_size']}/{dataset_info['test_size']}")
        print()
    except Exception as e:
        print(f"Warning: Could not load dataset info: {e}")
        print()
    
    # Генерація конфігурацій
    configs = get_standard_configs()
    # Add baseline "config" (string marker)
    configs_to_run = ['baseline'] + configs
    
    print(f"Running {len(configs_to_run)} configurations (including baseline)...")
    print()
    
    # Запуск експериментів
    all_results = {}
    experiment_start = time.time()
    
    for idx, config in enumerate(configs_to_run, 1):
        if config == 'baseline':
            config_name = 'baseline'
        else:
            config_name = config.get_name()
            
        print(f"[{idx}/{len(configs_to_run)}] Running: {config_name}")
        
        if args.verbose and config != 'baseline':
            print(f"  Config: {config.to_dict()}")
        
        try:
            results = run_multiple_runs(
                config,
                args.dataset,
                args.csv_path,
                args.target,
                args.n_runs,
                base_seed=args.seed
                ,
                deterministic=args.deterministic,
                forced_task_type=args.task_type
            )
            
            # Збереження результатів
            all_results[config_name] = results
            
            # Determine whether config is regression (use r2_score) or classification (use accuracy)
            sample_metric = 'accuracy'
            # If first run has 'r2_score' use that metric
            # If user forced the task type, prefer that metric type
            if args.task_type == 'regression':
                if 'r2_score' in results[0]:
                    sample_metric = 'r2_score'
                elif 'rmse' in results[0]:
                    sample_metric = 'rmse'
                else:
                    sample_metric = 'r2_score' if 'r2_score' in results[0] else 'rmse' if 'rmse' in results[0] else 'accuracy'
            elif args.task_type == 'classification':
                if 'accuracy' in results[0]:
                    sample_metric = 'accuracy'
                elif 'f1_score' in results[0]:
                    sample_metric = 'f1_score'
                else:
                    sample_metric = 'accuracy'
            else:
                if 'r2_score' in results[0]:
                    sample_metric = 'r2_score'
                elif 'rmse' in results[0]:
                    sample_metric = 'rmse'

            avg_val = np.mean([r[sample_metric] for r in results])
            avg_time = np.mean([r['train_time'] for r in results])

            if sample_metric == 'r2_score':
                print(f"  → R2: {avg_val:.4f}, Time: {avg_time:.2f}s")
            else:
                print(f"  → {sample_metric.upper()}: {avg_val:.4f}, Time: {avg_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        print()
    
    total_time = time.time() - experiment_start
    print(f"Total experiment time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print()
    
    # Збереження детальних результатів
    print("Saving detailed results...")
    detailed_results = []
    for config_name, results in all_results.items():
        for result in results:
            result['configuration'] = config_name
            detailed_results.append(result)
    
    df_detailed = pd.DataFrame(detailed_results)
    csv_path = output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_detailed.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    
    # Збереження підсумкової статистики
    print("\nCalculating summary statistics...")
    # Determine primary metric to use for statistical comparisons (prefer R2 for regression)
    metric_candidates = ['r2_score', 'accuracy', 'rmse']
    selected_metric = None
    # pick the first candidate that's present in results
    for candidate in metric_candidates:
        any_present = any(candidate in r for rs in all_results.values() for r in rs)
        if any_present:
            selected_metric = candidate
            break
    if selected_metric is None:
        selected_metric = 'accuracy'

    results_dict = {
        config_name: np.array([r[selected_metric] for r in results])
        for config_name, results in all_results.items()
    }
    
    summary_df = summarize_statistics(results_dict)
    summary_path = output_dir / f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  → {summary_path}")
    
    # Статистичний аналіз
    print("\nPerforming statistical analysis...")
    stat_report = generate_statistical_report(results_dict, alpha=0.05)
    print(stat_report)
    
    # Збереження статистичного звіту
    report_path = output_dir / f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(stat_report)
    print(f"Statistical report saved to: {report_path}")
    
    # Попарні порівняння
    if len(results_dict) >= 2:
        print("\nPerforming pairwise comparisons...")
        comparison_df = pairwise_comparison(results_dict, alpha=0.05)
        comparison_path = output_dir / f"pairwise_comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  → {comparison_path}")
    else:
        comparison_df = None
    
    # Візуалізація
    if not args.no_plots:
        print("\nGenerating plots...")
        try:
            # metric name used in plots
            metric_name = 'Accuracy' if selected_metric == 'accuracy' else ('R2' if selected_metric == 'r2_score' else selected_metric.upper())
            create_all_plots(
                results_dict,
                comparison_df,
                output_dir=str(output_dir),
                metric_name=metric_name
            )
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    # Підсумок
    print("\n" + "="*70)
    print("ABLATION ANALYSIS COMPLETED")
    print("="*70)
    print(f"\nBest configuration:")
    best_config = summary_df.iloc[0]
    print(f"  {best_config['configuration']}")
    print(f"  Mean {('R2' if selected_metric == 'r2_score' else selected_metric.upper())}: {best_config['mean']:.4f} ± {best_config['std']:.4f}")
    print(f"  95% CI: [{best_config['ci_lower']:.4f}, {best_config['ci_upper']:.4f}]")
    print()
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
