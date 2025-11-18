#!/usr/bin/env python
"""
Тестування різних моделей Gemini для генерації ML pipeline.
Порівнюємо якість згенерованого коду та точність pipeline.

Моделі для тестування:
- gemini-2.0-flash-exp (безкоштовна, швидка)
- gemini-1.5-flash (дешева, 0.075$ / 1M tokens)
- gemini-1.5-pro (дорожча, 1.25$ / 1M tokens, розумніша)
- gemini-2.0-flash-thinking-exp (новітня з reasoning)

Автор: Фефелов Ілля Олександрович
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import traceback

import google.generativeai as genai
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

# Додаємо src до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_pipeline_code(model_name: str, api_key: str, dataset_name: str = "breast_cancer") -> Tuple[str, Dict]:
    """
    Генерує ML pipeline через Gemini API.
    
    Args:
        model_name: Назва моделі Gemini
        api_key: API key
        dataset_name: Назва датасету
        
    Returns:
        Tuple[str, Dict]: (згенерований код, метаінформація)
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Отримання інфо про датасет
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        n_samples, n_features = data.data.shape
        n_classes = len(set(data.target))
    
    prompt = f"""
Generate a complete scikit-learn ML pipeline for the '{dataset_name}' dataset.

Dataset Information:
- Samples: {n_samples}
- Features: {n_features}
- Classes: {n_classes}
- Task: Binary Classification

Requirements:
1. Create a Pipeline with these steps:
   - 'preprocessor': Handle missing values and scaling (Pipeline with SimpleImputer + StandardScaler)
   - 'feature_engineering': Optional dimensionality reduction (e.g., PCA)
   - 'model': Classification model (LogisticRegression, RandomForest, SVC, etc.)

2. Return ONLY the Python function code that builds the pipeline:

```python
def build_full_pipeline(random_state: int = 42, numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None) -> Pipeline:
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
    
    print(f"\n🤖 Model: {model_name}")
    print("   Generating pipeline...", end=" ")
    
    start_time = time.time()
    try:
        # Deterministic generation to reduce variance across runs (fallback)
        try:
            response = model.generate_content(prompt, temperature=0, top_p=1)
        except TypeError:
            response = model.generate_content(prompt)
        generation_time = time.time() - start_time
        
        # Витягуємо код
        code = response.text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Підрахунок токенів (приблизно)
        estimated_tokens = len(prompt.split()) + len(code.split())
        
        metadata = {
            "model": model_name,
            "generation_time": generation_time,
            "code_length": len(code),
            "estimated_tokens": estimated_tokens,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✓ ({generation_time:.2f}s)")
        return code, metadata
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, {"model": model_name, "error": str(e)}


def test_pipeline_code(code: str, dataset_name: str = "breast_cancer") -> Dict:
    """
    Тестує згенерований pipeline код.
    
    Args:
        code: Python код з функцією build_full_pipeline()
        dataset_name: Назва датасету
        
    Returns:
        Dict: Результати тестування (accuracy, train_time, errors)
    """
    print("   Testing pipeline...", end=" ")
    
    try:
        # Виконуємо код у безпечному namespace
        namespace = {}
        exec(code, namespace)
        
        if 'build_full_pipeline' not in namespace:
            return {"error": "Function 'build_full_pipeline' not found in generated code"}
        
        build_full_pipeline = namespace['build_full_pipeline']
        
        # Завантажуємо дані
        if dataset_name == "breast_cancer":
            X, y = load_breast_cancer(return_X_y=True)
        
        # Створюємо pipeline
        pipeline = build_full_pipeline()
        
        # Тестуємо з cross-validation
        start_time = time.time()
        scores = cross_val_score(pipeline, X[:200], y[:200], cv=3, scoring='accuracy')
        train_time = time.time() - start_time
        
        print(f"✓ Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "train_time": train_time,
            "success": True,
            "pipeline_steps": [name for name, _ in pipeline.steps]
        }
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:100]}")
        return {
            "error": str(e),
            "success": False,
            "traceback": traceback.format_exc()
        }


def analyze_code_quality(code: str) -> Dict:
    """
    Аналізує якість згенерованого коду.
    
    Args:
        code: Python код
        
    Returns:
        Dict: Метрики якості коду
    """
    metrics = {
        "lines_of_code": len(code.split('\n')),
        "has_comments": '#' in code or '"""' in code or "'''" in code,
        "has_docstring": '"""' in code or "'''" in code,
        "has_imports": 'import' in code or 'from' in code,
        "has_pipeline": 'Pipeline' in code,
        "has_preprocessor": 'preprocessor' in code.lower(),
        "has_feature_engineering": 'feature' in code.lower(),
        "has_model": 'model' in code.lower() or 'classifier' in code.lower(),
        "has_random_state": 'random_state' in code,
    }
    
    # Підрахунок коментарів
    comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
    metrics["comment_lines"] = len(comment_lines)
    metrics["comment_ratio"] = len(comment_lines) / max(1, metrics["lines_of_code"])
    
    return metrics


def main():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set!")
        return
    
    # Моделі для тестування (доступні у вашому paid tier)
    models = [
        "gemini-2.5-flash-lite",             # Найшвидша (15 RPM, 3K TPM)
        "gemini-2.5-flash",                  # Збалансована (10 RPM, 1.48K TPM)
        "gemini-2.5-pro",                    # Найрозумніша (2 RPM, 394 TPM)
    ]
    
    print(f"\n💡 Testing with paid tier limits:")
    print(f"   - gemini-2.5-flash-lite: 15 RPM (fastest)")
    print(f"   - gemini-2.5-flash: 10 RPM (balanced)")
    print(f"   - gemini-2.5-pro: 2 RPM (smartest)")
    print(f"   Total test time: ~2-3 minutes with delays\n")
    
    print("="*80)
    print("GEMINI MODEL COMPARISON FOR ML PIPELINE GENERATION")
    print("="*80)
    print(f"Dataset: breast_cancer")
    print(f"Models to test: {len(models)}")
    print("="*80)
    
    results = []
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")
        
        # Генерація коду
        code, metadata = generate_pipeline_code(model_name, api_key)
        
        if code is None:
            results.append(metadata)
            continue
        
        # Аналіз якості коду
        code_quality = analyze_code_quality(code)
        print(f"   Code quality: {code_quality['lines_of_code']} lines, "
              f"{code_quality['comment_lines']} comments, "
              f"{'✓' if code_quality['has_docstring'] else '✗'} docstring")
        
        # Тестування pipeline
        test_results = test_pipeline_code(code)
        
        # Збираємо результати
        result = {
            **metadata,
            **code_quality,
            **test_results
        }
        results.append(result)
        
        # Зберігаємо код
        output_dir = Path("test_results_gemini_models")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        code_file = output_dir / f"pipeline_{model_name.replace('-', '_').replace('.', '_')}.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(f'"""\nGenerated by: {model_name}\n')
            f.write(f'Timestamp: {metadata["timestamp"]}\n')
            f.write(f'Generation time: {metadata["generation_time"]:.2f}s\n')
            f.write('"""\n\n')
            f.write(code)
        
        print(f"   Saved to: {code_file}")
        
        # Пауза між запитами (враховуємо різні ліміти)
        if model_name != models[-1]:  # Не чекаємо після останньої моделі
            # gemini-2.5-pro має найменший ліміт (2 RPM = 30s між запитами)
            wait_time = 35 if "pro" in model_name else 10
            print(f"   ⏳ Waiting {wait_time} seconds (rate limit)...")
            time.sleep(wait_time)
    
    # Підсумкова таблиця
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    # Заголовок таблиці
    print(f"{'Model':<35} {'Gen.Time':>10} {'Accuracy':>12} {'Code Lines':>12} {'Comments':>10}")
    print("-"*80)
    
    for result in results:
        model = result.get('model', 'Unknown')[:35]
        gen_time = result.get('generation_time', 0)
        accuracy = result.get('accuracy_mean', 0)
        lines = result.get('lines_of_code', 0)
        comments = result.get('comment_lines', 0)
        
        if result.get('success'):
            print(f"{model:<35} {gen_time:>9.2f}s "
                  f"{accuracy:>11.4f} "
                  f"{lines:>12} "
                  f"{comments:>10}")
        else:
            print(f"{model:<35} {gen_time:>9.2f}s "
                  f"{'ERROR':>12} "
                  f"{lines:>12} "
                  f"{comments:>10}")
    
    # Збереження JSON результатів
    output_dir = Path("test_results_gemini_models")
    output_dir.mkdir(exist_ok=True, parents=True)
    json_file = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {json_file}")
    print(f"{'='*80}\n")
    
    # Рекомендації
    print("RECOMMENDATIONS:")
    print("-"*80)
    
    successful_results = [r for r in results if r.get('success')]
    if successful_results:
        # Найточніша модель
        best_accuracy = max(successful_results, key=lambda x: x.get('accuracy_mean', 0))
        print(f"🎯 Best accuracy: {best_accuracy['model']}")
        print(f"   Accuracy: {best_accuracy['accuracy_mean']:.4f}")
        
        # Найшвидша модель
        fastest = min(successful_results, key=lambda x: x.get('generation_time', float('inf')))
        print(f"\n⚡ Fastest: {fastest['model']}")
        print(f"   Time: {fastest['generation_time']:.2f}s")
        
        # Найдокументованіша
        best_documented = max(successful_results, key=lambda x: x.get('comment_ratio', 0))
        print(f"\n📝 Best documented: {best_documented['model']}")
        print(f"   Comments: {best_documented['comment_lines']} lines "
              f"({best_documented['comment_ratio']*100:.1f}%)")
        
        # Рекомендація для диплому
        print(f"\n💡 RECOMMENDATION FOR DIPLOMA:")
        
        # Якщо різниця в accuracy < 1%, використовуємо дешевшу
        accuracies = [r['accuracy_mean'] for r in successful_results]
        accuracy_range = max(accuracies) - min(accuracies)
        
        if accuracy_range < 0.01:
            print("   All models show similar accuracy (<1% difference).")
            print(f"   ✅ Use fastest/cheapest: gemini-2.0-flash-exp or gemini-1.5-flash")
            print("   💰 Cost: FREE (2.0-flash-exp) or $0.075/1M tokens (1.5-flash)")
        else:
            print(f"   Accuracy varies by {accuracy_range*100:.2f}%")
            if best_accuracy['accuracy_mean'] - min(accuracies) > 0.02:
                print(f"   ✅ Use best performing: {best_accuracy['model']}")
                print(f"   Significant improvement justifies cost")
            else:
                print(f"   ✅ Use balanced: gemini-1.5-flash")
                print(f"   Good balance between cost and quality")
    
    print("="*80)


if __name__ == "__main__":
    main()
