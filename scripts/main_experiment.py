#!/usr/bin/env python
"""
Головний скрипт для автоматизованого проведення експериментів:
1. Генерація ML pipeline через Gemini API для кожного датасету
2. Збереження згенерованого коду
3. Проведення ablation аналізу
4. Збір результатів для всіх датасетів

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import google.generativeai as genai
import numpy as np
import pandas as pd

# Додаємо src до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mle_star_ablation.config import get_standard_configs
from src.mle_star_ablation.datasets import DatasetLoader
from src.mle_star_ablation.metrics import calculate_classification_metrics


def generate_pipeline_with_gemini(dataset_name: str, api_key: str) -> str:
    """
    Генерує ML pipeline через Gemini API для заданого датасету.
    
    Args:
        dataset_name: Назва датасету (breast_cancer, wine, digits, iris)
        api_key: Gemini API key
        
    Returns:
        str: Згенерований Python код pipeline
    """
    # Конфігурація Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    # Отримання інформації про датасет
    dataset_info = DatasetLoader.get_dataset_info(dataset_name)
    
    # Промпт для генерації
    prompt = f"""
Generate a complete scikit-learn ML pipeline for the '{dataset_name}' dataset.

Dataset Information:
- Samples: {dataset_info['n_samples']}
- Features: {dataset_info['n_features']}
- Classes: {dataset_info['n_classes']}
- Task: Classification

Requirements:
1. Create a Pipeline with these steps:
   - 'preprocessor': Handle missing values and scaling (Pipeline with SimpleImputer + StandardScaler)
   - 'feature_engineering': Optional dimensionality reduction (e.g., PCA)
   - 'model': Classification model (LogisticRegression, RandomForest, SVC, etc.)

2. Return ONLY the Python function code that builds the pipeline like:

```python
def build_full_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    # ... other imports
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    feature_engineering = Pipeline([
        # your feature engineering steps
    ])
    
    model = # your model choice
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])
```

3. Add detailed comments explaining:
   - Why you chose each component
   - What hyperparameters you selected and why
   - How each step helps for this specific dataset

4. Use random_state=42 where applicable
5. Choose appropriate model and hyperparameters for this dataset size and complexity
6. Keep preprocessing simple but effective

Generate ONLY the function code, no explanations outside the code.
"""
    
    print(f"\n📡 Generating pipeline for {dataset_name} via Gemini API...")
    response = model.generate_content(prompt)
    
    # Витягуємо код з відповіді
    code = response.text
    
    # Якщо код обгорнутий у ```python, розпаковуємо
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    return code


def save_generated_pipeline(code: str, dataset_name: str, output_dir: Path):
    """
    Зберігає згенерований код у файл.
    
    Args:
        code: Згенерований Python код
        dataset_name: Назва датасету
        output_dir: Директорія для збереження
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f"pipeline_{dataset_name}_{timestamp}.py"
    
    # Формуємо повний файл з header
    full_code = f'''"""
Generated ML Pipeline for {dataset_name} dataset
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generator: Google Gemini 2.0 Flash Exp
"""

{code}


if __name__ == "__main__":
    # Test the pipeline
    from sklearn.datasets import load_{dataset_name}
    from sklearn.model_selection import cross_val_score
    
    X, y = load_{dataset_name}(return_X_y=True)
    pipeline = build_full_pipeline()
    
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    print(f"Pipeline for {dataset_name}:")
    print(f"  Accuracy: {{scores.mean():.3f}} ± {{scores.std():.3f}}")
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print(f"✅ Saved to {filename}")
    return filename


def update_mle_star_pipeline(code: str, dataset_name: str):
    """
    Оновлює mle_star_generated_pipeline.py новим кодом.
    
    Args:
        code: Згенерований код pipeline
        dataset_name: Назва датасету (для документації)
    """
    pipeline_file = Path(__file__).parent.parent / "src" / "mle_star_ablation" / "mle_star_generated_pipeline.py"
    
    # Читаємо поточний файл
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Знаходимо секцію build_full_pipeline
    start_marker = "def build_full_pipeline("
    end_marker = "\n\n# ==== 2. НАЗВИ КРОКІВ"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find build_full_pipeline section in the file")
    
    # Формуємо новий код з правильною сигнатурою
    new_function = f'''def build_full_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Повертає повний ML-конвеєр, згенерований Gemini API для {dataset_name}.
    
    Pipeline складається з:
    1. Preprocessor: SimpleImputer + StandardScaler
    2. Feature Engineering: (визначено Gemini)
    3. Model: (визначено Gemini)
    
    Згенеровано: Google Gemini 2.0 Flash Exp
    Датасет: {dataset_name}
    Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Returns:
        Pipeline: Повний sklearn Pipeline з усіма компонентами
    """
{code.replace("def build_full_pipeline():", "").strip()}
'''
    
    # Замінюємо старий код
    new_content = content[:start_idx] + new_function + content[end_idx:]
    
    # Зберігаємо
    with open(pipeline_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Updated {pipeline_file}")


def run_ablation_for_dataset(dataset_name: str, n_runs: int = 5) -> Dict:
    """
    Запускає ablation аналіз для одного датасету.
    
    Args:
        dataset_name: Назва датасету
        n_runs: Кількість повторів кожної конфігурації
        
    Returns:
        Dict: Результати всіх конфігурацій
    """
    from scripts.run_ablation import run_multiple_runs
    
    configs = get_standard_configs()
    results = {}
    
    print(f"\n🔬 Running ablation analysis for {dataset_name}...")
    print(f"   Configurations: {len(configs)}")
    print(f"   Runs per config: {n_runs}")
    
    for idx, config in enumerate(configs, 1):
        config_name = config.get_name()
        print(f"   [{idx}/{len(configs)}] {config_name}...", end=" ")
        
        try:
            config_results = run_multiple_runs(
                config, dataset_name, None, None, n_runs
            )
            results[config_name] = config_results
            
            avg_acc = np.mean([r['accuracy'] for r in config_results])
            print(f"✓ {avg_acc:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results[config_name] = []
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Автоматизована генерація та ablation аналіз для всіх датасетів',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['breast_cancer', 'wine', 'digits', 'iris'],
        help='Список датасетів для аналізу'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=5,
        help='Кількість повторів кожної конфігурації'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_full_experiment',
        help='Директорія для збереження результатів'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Пропустити генерацію pipeline (використати існуючий)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Gemini API key (або через GEMINI_API_KEY env)'
    )
    
    args = parser.parse_args()
    
    # API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key and not args.skip_generation:
        raise ValueError("Gemini API key required! Set --api-key or GEMINI_API_KEY env")
    
    # Створюємо директорії
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    generated_dir = Path("generated_pipelines")
    
    print("="*80)
    print("AUTOMATED ABLATION EXPERIMENT")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Runs per config: {args.n_runs}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Результати для всіх датасетів
    all_dataset_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # 1. Генерація pipeline (якщо потрібно)
        if not args.skip_generation:
            try:
                code = generate_pipeline_with_gemini(dataset_name, api_key)
                
                # Зберігаємо згенерований код
                save_generated_pipeline(code, dataset_name, generated_dir)
                
                # Оновлюємо mle_star_generated_pipeline.py
                update_mle_star_pipeline(code, dataset_name)
                
                print("✅ Pipeline generated and installed")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                continue
        else:
            print("⏭️  Skipping generation (using existing pipeline)")
        
        # 2. Запуск ablation аналізу
        try:
            results = run_ablation_for_dataset(dataset_name, args.n_runs)
            all_dataset_results[dataset_name] = results
            
            # Збереження результатів для цього датасету
            dataset_output = output_dir / dataset_name
            dataset_output.mkdir(exist_ok=True, parents=True)
            
            # Детальні результати
            detailed_results = []
            for config_name, config_results in results.items():
                for result in config_results:
                    result['dataset'] = dataset_name
                    result['configuration'] = config_name
                    detailed_results.append(result)
            
            df = pd.DataFrame(detailed_results)
            csv_path = dataset_output / f"results_{dataset_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Results saved to {csv_path}")
            
        except Exception as e:
            print(f"❌ Ablation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Збір підсумкових результатів
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    summary_data = []
    for dataset_name, results in all_dataset_results.items():
        for config_name, config_results in results.items():
            if config_results:
                accuracies = [r['accuracy'] for r in config_results]
                summary_data.append({
                    'dataset': dataset_name,
                    'configuration': config_name,
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'n_runs': len(accuracies)
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "summary_all_datasets.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n📊 Summary statistics:")
    print(summary_df.pivot_table(
        values='mean_accuracy',
        index='configuration',
        columns='dataset',
        aggfunc='mean'
    ))
    
    print(f"\n✅ All results saved to: {args.output_dir}")
    print(f"   Summary: {summary_path}")
    print("="*80)


if __name__ == "__main__":
    main()
