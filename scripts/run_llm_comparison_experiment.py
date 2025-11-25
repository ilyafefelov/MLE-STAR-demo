#!/usr/bin/env python3
"""
Експеримент порівняння LLM: Simple Prompt vs MLE-STAR vs GPT-4o
================================================================

Мета: Перевірити, чи гіпотеза over-engineering залежить від:
1. Складності промпту (simple vs MLE-STAR)
2. Вибору LLM (Gemini vs GPT-4o)

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

# Завантаження API ключів
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Директорії
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ПРОМПТИ
# ============================================================================

SIMPLE_PROMPT_TEMPLATE = """
Створи sklearn Pipeline для {task_type} на датасеті {dataset_name}.

Дані: {n_samples} зразків, {n_features} ознак{class_info}.
ВАЖЛИВО: Дані передаються як numpy array (X, y), НЕ pandas DataFrame!

Поверни ТІЛЬКИ Python код:
- Усі імпорти НА ПОЧАТКУ файлу (перед def)
- НЕ використовуй type hints у сигнатурі функції
- Простий pipeline: StandardScaler + модель (максимум 2-3 кроки)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # або інша модель

def build_full_pipeline(random_state=42):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(random_state=random_state))
    ])
    return pipeline
```
"""

MLE_STAR_PROMPT_TEMPLATE = """
Ти експерт Machine Learning Engineer. Генеруй повний scikit-learn ML pipeline.

ДАТАСЕТ: {dataset_name}
- Зразків: {n_samples}
- Ознак: {n_features}
{class_info}

ВИМОГИ:
1. Pipeline має містити:
   - preprocessor: SimpleImputer + StandardScaler
   - feature_engineering: PCA або PolynomialFeatures
   - model: Обери НАЙКРАЩУ модель (RandomForest, GradientBoosting, SVM, etc.)

2. Використовуй ансамблювання (VotingClassifier/VotingRegressor) якщо це покращить результат

3. Налаштуй гіперпараметри для цього конкретного датасету

4. Додай коментарі з обґрунтуванням вибору

5. ВАЖЛИВО: Дані передаються як numpy array (X, y), НЕ DataFrame. 
   НЕ використовуй ColumnTransformer з іменами колонок!
   Працюй з усіма ознаками одразу (StandardScaler для всіх).

6. ЗАБОРОНЕНО: GridSearchCV, RandomizedSearchCV (занадто повільно)
   Використовуй фіксовані гіперпараметри.

ФОРМАТ ВІДПОВІДІ - ТІЛЬКИ КОД:
- Усі імпорти НА ПОЧАТКУ файлу
- НЕ використовуй type hints (-> Pipeline)
- Функція повертає sklearn.pipeline.Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# ... інші імпорти

def build_full_pipeline(random_state=42):
    # твій код
    return pipeline
```
"""


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Завантажує датасет і повертає його інформацію."""
    if dataset_name == "iris":
        data = load_iris()
        task_type = "класифікації"
        class_info = f", {len(np.unique(data.target))} класів"
        is_classification = True
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        task_type = "класифікації"
        class_info = f", {len(np.unique(data.target))} класів"
        is_classification = True
    elif dataset_name == "california_housing":
        data = fetch_california_housing()
        task_type = "регресії"
        class_info = ""
        is_classification = False
    else:
        raise ValueError(f"Невідомий датасет: {dataset_name}")
    
    return {
        "X": data.data,
        "y": data.target,
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "task_type": task_type,
        "class_info": class_info,
        "is_classification": is_classification,
        "dataset_name": dataset_name,
    }


def format_prompt(template: str, info: Dict[str, Any]) -> str:
    """Форматує промпт з інформацією про датасет."""
    return template.format(
        dataset_name=info["dataset_name"],
        n_samples=info["n_samples"],
        n_features=info["n_features"],
        task_type=info["task_type"],
        class_info=info["class_info"],
    )


# ============================================================================
# LLM API ВИКЛИКИ
# ============================================================================

def call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Виклик Gemini API."""
    import google.generativeai as genai
    
    genai.configure(api_key=GEMINI_API_KEY)
    model_obj = genai.GenerativeModel(model)
    
    response = model_obj.generate_content(prompt)
    return response.text


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Виклик OpenAI API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ти експерт Machine Learning Engineer. Відповідай ТІЛЬКИ Python кодом."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def extract_code(response: str) -> str:
    """Витягує Python код з відповіді LLM."""
    # Шукаємо код між ```python і ```
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Якщо немає маркерів, шукаємо def build_full_pipeline
    if "def build_full_pipeline" in response:
        start = response.find("def build_full_pipeline")
        # Знаходимо кінець функції (наступний def або кінець)
        end = response.find("\ndef ", start + 1)
        if end == -1:
            end = len(response)
        return response[start:end].strip()
    
    return response


def execute_code_and_get_pipeline(code: str, random_state: int = 42, timeout_sec: int = 30) -> Optional[Pipeline]:
    """Виконує код і повертає Pipeline з timeout."""
    import signal
    import threading
    
    # Створюємо namespace для виконання
    namespace = {}
    result = {"pipeline": None, "error": None}
    
    def run_code():
        try:
            exec(code, namespace)
            
            if "build_full_pipeline" not in namespace:
                result["error"] = "Функція build_full_pipeline не знайдена"
                return
            
            result["pipeline"] = namespace["build_full_pipeline"](random_state=random_state)
        except Exception as e:
            result["error"] = str(e)
    
    # Запускаємо в окремому потоці з timeout
    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        print(f"  ⚠️ Timeout ({timeout_sec}с) - код виконується занадто довго")
        return None
    
    if result["error"]:
        print(f"  ⚠️ Помилка виконання коду: {result['error']}")
        return None
    
    return result["pipeline"]


def evaluate_pipeline(
    pipeline: Pipeline, 
    X: np.ndarray, 
    y: np.ndarray, 
    is_classification: bool,
    n_runs: int = 5,
) -> Dict[str, float]:
    """Оцінює pipeline за допомогою cross-validation."""
    scores = []
    
    for run in range(n_runs):
        seed = 42 + run
        
        if is_classification:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scoring = "accuracy"
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            scoring = "r2"
        
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            scores.append(np.mean(cv_scores))
        except Exception as e:
            print(f"  ⚠️ Помилка CV (run {run}): {e}")
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "n_runs": 0}
    
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "n_runs": len(scores),
    }


# ============================================================================
# ГОЛОВНИЙ ЕКСПЕРИМЕНТ
# ============================================================================

def run_single_experiment(
    dataset_name: str,
    prompt_type: str,  # "simple" або "mle_star"
    llm_type: str,     # "gemini" або "gpt4o"
    n_runs: int = 5,
) -> Dict[str, Any]:
    """Запускає один експеримент."""
    print(f"\n{'='*60}")
    print(f"Датасет: {dataset_name} | Промпт: {prompt_type} | LLM: {llm_type}")
    print(f"{'='*60}")
    
    # Завантажуємо датасет
    info = get_dataset_info(dataset_name)
    
    # Вибираємо шаблон промпту
    if prompt_type == "simple":
        template = SIMPLE_PROMPT_TEMPLATE
    else:
        template = MLE_STAR_PROMPT_TEMPLATE
    
    prompt = format_prompt(template, info)
    
    # Виклик LLM
    print(f"📤 Відправляю запит до {llm_type}...")
    start_time = time.time()
    
    try:
        if llm_type == "gemini":
            response = call_gemini(prompt)
        else:
            response = call_openai(prompt)
        
        api_time = time.time() - start_time
        print(f"✅ Відповідь отримано за {api_time:.1f}с")
        
    except Exception as e:
        print(f"❌ Помилка API: {e}")
        return {
            "dataset": dataset_name,
            "prompt_type": prompt_type,
            "llm_type": llm_type,
            "status": "api_error",
            "error": str(e),
        }
    
    # Витягуємо код
    code = extract_code(response)
    
    # Виконуємо код
    print("🔧 Будую pipeline...")
    pipeline = execute_code_and_get_pipeline(code)
    
    if pipeline is None:
        return {
            "dataset": dataset_name,
            "prompt_type": prompt_type,
            "llm_type": llm_type,
            "status": "code_error",
            "code": code,
            "response": response[:500],
        }
    
    # Оцінюємо
    print(f"📊 Оцінюю pipeline ({n_runs} запусків)...")
    metrics = evaluate_pipeline(
        pipeline, 
        info["X"], 
        info["y"], 
        info["is_classification"],
        n_runs=n_runs,
    )
    
    print(f"✅ Результат: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
    
    # Аналіз складності pipeline
    pipeline_info = analyze_pipeline_complexity(pipeline)
    
    return {
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "llm_type": llm_type,
        "status": "success",
        "score_mean": metrics["mean"],
        "score_std": metrics["std"],
        "n_runs": metrics["n_runs"],
        "api_time_sec": api_time,
        **pipeline_info,
    }


def analyze_pipeline_complexity(pipeline: Pipeline) -> Dict[str, Any]:
    """Аналізує складність pipeline."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import VotingClassifier, VotingRegressor, RandomForestClassifier
    
    steps = pipeline.steps
    n_steps = len(steps)
    
    # Визначаємо компоненти
    has_scaler = False
    has_pca = False
    has_ensemble = False
    model_type = "unknown"
    
    for name, est in steps:
        if isinstance(est, Pipeline):
            for _, nested in est.steps:
                if isinstance(nested, StandardScaler):
                    has_scaler = True
                if isinstance(nested, PCA):
                    has_pca = True
        else:
            if isinstance(est, StandardScaler):
                has_scaler = True
            if isinstance(est, PCA):
                has_pca = True
            if isinstance(est, (VotingClassifier, VotingRegressor)):
                has_ensemble = True
            # Останній крок - модель
            if name in ["model", "estimator", "clf", "regressor"] or steps[-1] == (name, est):
                model_type = type(est).__name__
    
    return {
        "n_steps": n_steps,
        "has_scaler": has_scaler,
        "has_pca": has_pca,
        "has_ensemble": has_ensemble,
        "model_type": model_type,
    }


def run_full_experiment():
    """Запускає повний експеримент."""
    print("\n" + "="*70)
    print("🧪 ЕКСПЕРИМЕНТ: ПОРІВНЯННЯ LLM ТА ТИПІВ ПРОМПТІВ")
    print("="*70)
    
    # Перевіряємо наявність API ключів
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_TOKEN не знайдено!")
        return
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY не знайдено - GPT-4o буде пропущено")
    
    datasets = ["iris", "breast_cancer", "california_housing"]
    prompt_types = ["simple", "mle_star"]
    llm_types = ["gemini"]
    if OPENAI_API_KEY:
        llm_types.append("gpt4o")
    
    results = []
    
    for dataset in datasets:
        for prompt_type in prompt_types:
            for llm_type in llm_types:
                result = run_single_experiment(
                    dataset_name=dataset,
                    prompt_type=prompt_type,
                    llm_type=llm_type,
                    n_runs=5,
                )
                results.append(result)
                
                # Невелика пауза між запитами
                time.sleep(1)
    
    # Зберігаємо результати
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = RESULTS_DIR / f"llm_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Результати збережено: {csv_path}")
    
    # Виводимо зведену таблицю
    print("\n" + "="*70)
    print("📊 ЗВЕДЕНІ РЕЗУЛЬТАТИ")
    print("="*70)
    
    success_df = df[df["status"] == "success"]
    if len(success_df) > 0:
        summary = success_df.groupby(["dataset", "prompt_type", "llm_type"]).agg({
            "score_mean": "mean",
            "score_std": "mean",
            "n_steps": "mean",
            "has_pca": "mean",
            "has_ensemble": "mean",
        }).round(4)
        print(summary.to_string())
    
    return df


if __name__ == "__main__":
    run_full_experiment()
