#!/usr/bin/env python
"""
Порівняння різних моделей Gemini на всіх датасетах.
Генеруємо pipeline для кожної комбінації (модель × датасет) та порівнюємо результати.

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import google.generativeai as genai
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import cross_val_score

# Додаємо src до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mle_star_ablation.datasets import DatasetLoader
from src.mle_star_ablation.prompts import generate_mle_prompt


# Моделі для порівняння
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",    # Найшвидша
    "gemini-2.5-flash",          # Збалансована
    "gemini-2.5-pro",            # Найрозумніша
]

# Датасети для тестування
DATASETS = ["breast_cancer", "wine", "digits", "iris"]


def generate_pipeline_code(model_name: str, dataset_name: str, api_key: str) -> Tuple[str, Dict]:
    """Генерує ML pipeline через Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    dataset_info = DatasetLoader.get_dataset_info(dataset_name)
    
    prompt = generate_mle_prompt(dataset_name, dataset_info)
    
    start_time = time.time()
    # Deterministic generation parameters if supported (fallback)
    try:
        response = model.generate_content(prompt, temperature=0, top_p=1)
    except TypeError:
        response = model.generate_content(prompt)
    generation_time = time.time() - start_time
    
    code = response.text
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "generation_time": generation_time,
        "code_length": len(code),
        "timestamp": datetime.now().isoformat()
    }
    
    return code, metadata


def test_pipeline(code: str, dataset_name: str) -> Dict:
    """Тестує згенерований pipeline."""
    try:
        # Виконуємо код
        namespace = {}
        exec(code, namespace)
        
        if 'build_full_pipeline' not in namespace:
            return {"error": "Function 'build_full_pipeline' not found", "success": False}
        
        build_full_pipeline = namespace['build_full_pipeline']
        
        # Завантажуємо дані
        if dataset_name == "breast_cancer":
            X, y = load_breast_cancer(return_X_y=True)
        elif dataset_name == "wine":
            X, y = load_wine(return_X_y=True)
        elif dataset_name == "digits":
            X, y = load_digits(return_X_y=True)
        elif dataset_name == "iris":
            X, y = load_iris(return_X_y=True)
        
        # Створюємо pipeline
        pipeline = build_full_pipeline()
        
        # Тестуємо з full cross-validation
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        
        # Визначаємо модель
        model_name = type(pipeline.named_steps['model']).__name__
        
        return {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "accuracy_min": float(scores.min()),
            "accuracy_max": float(scores.max()),
            "model_used": model_name,
            "pipeline_steps": [name for name, _ in pipeline.steps],
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def analyze_model_choice(code: str) -> Dict:
    """Аналізує який алгоритм обрала модель."""
    model_keywords = {
        "LogisticRegression": "Logistic Regression",
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "SVC": "Support Vector Machine",
        "MLPClassifier": "Neural Network (MLP)",
        "DecisionTree": "Decision Tree",
        "KNeighbors": "K-Nearest Neighbors",
        "GaussianNB": "Naive Bayes",
    }
    
    detected_models = []
    for keyword, full_name in model_keywords.items():
        if keyword in code:
            detected_models.append(full_name)
    
    return {
        "detected_models": detected_models,
        "complexity": "complex" if any(m in detected_models for m in 
            ["Random Forest", "Gradient Boosting", "Neural Network (MLP)"]) else "simple"
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, help='Gemini API Key')
    parser.add_argument('--datasets', nargs='*', default=None, help='List of datasets to run (default: all)')
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        # Try loading from .env
        from pathlib import Path
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'GEMINI_API_KEY=' in line:
                        api_key = line.strip().split('=', 1)[1]
                        os.environ['GEMINI_API_KEY'] = api_key
                        break
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set!")
        print("Usage: python compare_gemini_models_on_datasets.py --api-key YOUR_KEY")
        return
    
    datasets_to_run = args.datasets if args.datasets else DATASETS
    
    output_dir = Path("model_comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("COMPREHENSIVE GEMINI MODEL COMPARISON")
    print("="*80)
    print(f"Models: {', '.join(GEMINI_MODELS)}")
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Total combinations: {len(GEMINI_MODELS)} × {len(datasets_to_run)} = {len(GEMINI_MODELS) * len(datasets_to_run)}")
    print("="*80)
    
    all_results = []
    total_experiments = len(GEMINI_MODELS) * len(datasets_to_run)
    experiment_num = 0
    
    for dataset_name in datasets_to_run:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        for model_name in GEMINI_MODELS:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Testing: {model_name}")
            
            # Генерація pipeline
            print(f"   🤖 Generating pipeline...", end=" ")
            try:
                code, metadata = generate_pipeline_code(model_name, dataset_name, api_key)
                print(f"✓ ({metadata['generation_time']:.2f}s)")
                
                # Аналіз вибору моделі
                model_analysis = analyze_model_choice(code)
                print(f"   🔍 Detected models: {', '.join(model_analysis['detected_models'])}")
                
                # Тестування pipeline
                print(f"   🧪 Testing pipeline...", end=" ")
                test_results = test_pipeline(code, dataset_name)
                
                if test_results['success']:
                    print(f"✓ Accuracy: {test_results['accuracy_mean']:.4f} ± {test_results['accuracy_std']:.4f}")
                    print(f"   📊 Model used: {test_results['model_used']}")
                else:
                    print(f"✗ Error: {test_results['error'][:80]}")
                
                # Збираємо результати
                result = {
                    **metadata,
                    **test_results,
                    **model_analysis
                }
                all_results.append(result)
                
                # Зберігаємо код
                code_file = output_dir / f"{model_name.replace('-', '_')}_{dataset_name}.py"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""\nModel: {model_name}\nDataset: {dataset_name}\n')
                    f.write(f'Generated: {metadata["timestamp"]}\n"""\n\n{code}')
                
                # Пауза для rate limit
                if experiment_num < total_experiments:
                    wait_time = 35 if "pro" in model_name else 5
                    print(f"   ⏳ Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"✗ Error: {str(e)[:100]}")
                all_results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "error": str(e),
                    "success": False
                })
    
    # Збереження результатів
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # JSON
    json_file = output_dir / f"comparison_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON: {json_file}")
    
    # CSV
    df = pd.DataFrame(all_results)
    csv_file = output_dir / f"comparison_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV: {csv_file}")
    
    # Підсумкова таблиця
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}\n")
    
    # Успішні експерименти
    successful = df[df['success'] == True]
    
    if len(successful) > 0:
        # Таблиця accuracy по датасетам
        print("Accuracy by Dataset and Model:")
        print("-"*80)
        pivot = successful.pivot_table(
            values='accuracy_mean',
            index='model',
            columns='dataset',
            aggfunc='mean'
        )
        print(pivot.to_string())
        
        # Середня accuracy по моделях
        print(f"\n\nAverage Accuracy by Model:")
        print("-"*80)
        avg_by_model = successful.groupby('model')['accuracy_mean'].agg(['mean', 'std', 'count'])
        print(avg_by_model.to_string())
        
        # Середня швидкість по моделях
        print(f"\n\nGeneration Time by Model:")
        print("-"*80)
        time_by_model = successful.groupby('model')['generation_time'].agg(['mean', 'min', 'max'])
        print(time_by_model.to_string())
        
        # Використані алгоритми
        print(f"\n\nModel Algorithms Used:")
        print("-"*80)
        algo_counts = successful['model_used'].value_counts()
        print(algo_counts.to_string())
        
        # Найкраща модель для кожного датасету
        print(f"\n\nBest Model per Dataset:")
        print("-"*80)
        best_per_dataset = successful.loc[successful.groupby('dataset')['accuracy_mean'].idxmax()]
        for _, row in best_per_dataset.iterrows():
            print(f"  {row['dataset']:15s} → {row['model']:25s} "
                  f"({row['accuracy_mean']:.4f}, {row['model_used']})")
        
        # Рекомендації
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        best_overall = successful.loc[successful['accuracy_mean'].idxmax()]
        fastest = successful.loc[successful['generation_time'].idxmin()]
        
        print(f"\n🎯 Best Overall Performance:")
        print(f"   Model: {best_overall['model']}")
        print(f"   Dataset: {best_overall['dataset']}")
        print(f"   Accuracy: {best_overall['accuracy_mean']:.4f}")
        print(f"   Algorithm: {best_overall['model_used']}")
        
        print(f"\n⚡ Fastest Generation:")
        print(f"   Model: {fastest['model']}")
        print(f"   Time: {fastest['generation_time']:.2f}s")
        print(f"   Accuracy: {fastest['accuracy_mean']:.4f}")
        
        # Аналіз складності алгоритмів
        complex_count = len(successful[successful['complexity'] == 'complex'])
        simple_count = len(successful[successful['complexity'] == 'simple'])
        
        print(f"\n🧠 Algorithm Complexity:")
        print(f"   Complex models (RF/GB/MLP): {complex_count}/{len(successful)} ({complex_count/len(successful)*100:.1f}%)")
        print(f"   Simple models (LR): {simple_count}/{len(successful)} ({simple_count/len(successful)*100:.1f}%)")
        
        if complex_count > 0:
            complex_acc = successful[successful['complexity'] == 'complex']['accuracy_mean'].mean()
            simple_acc = successful[successful['complexity'] == 'simple']['accuracy_mean'].mean() if simple_count > 0 else 0
            print(f"   Avg accuracy (complex): {complex_acc:.4f}")
            if simple_count > 0:
                print(f"   Avg accuracy (simple): {simple_acc:.4f}")
                print(f"   Difference: {complex_acc - simple_acc:+.4f}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
