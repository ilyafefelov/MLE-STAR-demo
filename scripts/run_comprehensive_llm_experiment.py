#!/usr/bin/env python3
"""
Комплексний експеримент: Simple vs MLE-STAR Prompt vs ADK MLE-STAR Agent
=========================================================================

Порівняння трьох підходів до генерації ML pipelines:
1. Simple Prompt (Gemini / GPT-4o) - мінімальні інструкції
2. MLE-STAR Prompt (Gemini / GPT-4o) - детальні інструкції з MLE-STAR
3. ADK MLE-STAR Agent - повноцінний multi-agent system

"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Завантаження API ключів
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Директорії
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_comparison"
GENERATED_PIPELINES_DIR = PROJECT_ROOT / "generated_pipelines"
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


# ============================================================================
# ДАТАСЕТИ
# ============================================================================

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


# ============================================================================
# ВИКОНАННЯ КОДУ
# ============================================================================

def execute_code_and_get_pipeline(code: str, random_state: int = 42, timeout_sec: int = 30) -> Optional[Pipeline]:
    """Виконує код і повертає Pipeline з timeout."""
    import threading
    
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


def extract_model_from_adk_script(code: str) -> Optional[Any]:
    """
    Витягує модель з ADK MLE-STAR скрипту.
    ADK генерує повні скрипти з train/predict, а не функції build_full_pipeline.
    """
    import threading
    
    # Шукаємо ініціалізацію моделі в коді
    # Типові патерни: model = ..., clf = ..., gbdt = ...
    
    namespace = {
        'pd': pd,
        'np': np,
    }
    result = {"model": None, "error": None}
    
    def run_code():
        try:
            # Імпортуємо всі потрібні бібліотеки
            exec("import pandas as pd", namespace)
            exec("import numpy as np", namespace)
            exec("from sklearn.ensemble import *", namespace)
            exec("from sklearn.linear_model import *", namespace)
            exec("from sklearn.svm import *", namespace)
            exec("from sklearn.tree import *", namespace)
            exec("from sklearn.preprocessing import StandardScaler", namespace)
            exec("from sklearn.impute import SimpleImputer", namespace)
            exec("from sklearn.pipeline import Pipeline", namespace)
            
            # Модифікуємо код: прибираємо завантаження даних та fit
            modified_code = code
            
            # Прибираємо рядки з read_csv та fit
            lines = modified_code.split('\n')
            filtered_lines = []
            skip_until_blank = False
            
            for line in lines:
                # Пропускаємо завантаження даних
                if 'read_csv' in line or '.fit(' in line or 'train_test_split' in line:
                    continue
                if 'X_train' in line or 'X_val' in line or 'y_train' in line or 'y_val' in line:
                    continue
                if 'predict' in line or 'accuracy' in line or 'rmse' in line:
                    continue
                if 'print(' in line:
                    continue
                filtered_lines.append(line)
            
            modified_code = '\n'.join(filtered_lines)
            
            exec(modified_code, namespace)
            
            # Шукаємо модель
            model_names = ['model', 'clf', 'gbdt', 'regressor', 'classifier', 'lgb_model']
            for name in model_names:
                if name in namespace:
                    result["model"] = namespace[name]
                    return
            
            result["error"] = "Модель не знайдена в коді"
            
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join(timeout=10)
    
    if thread.is_alive():
        return None
    
    if result["error"]:
        print(f"  ⚠️ Помилка ADK: {result['error']}")
        return None
    
    return result["model"]


def build_pipeline_from_model(model: Any, add_preprocessing: bool = True) -> Pipeline:
    """Створює Pipeline з моделі."""
    if add_preprocessing:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        return Pipeline([('model', model)])


# ============================================================================
# ОЦІНЮВАННЯ
# ============================================================================

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


def analyze_pipeline_complexity(pipeline: Pipeline) -> Dict[str, Any]:
    """Аналізує складність pipeline."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    
    steps = pipeline.steps
    n_steps = len(steps)
    
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
            if name in ["model", "estimator", "clf", "regressor"] or steps[-1] == (name, est):
                model_type = type(est).__name__
    
    return {
        "n_steps": n_steps,
        "has_scaler": has_scaler,
        "has_pca": has_pca,
        "has_ensemble": has_ensemble,
        "model_type": model_type,
    }


# ============================================================================
# ADK MLE-STAR PIPELINES
# ============================================================================

def get_adk_pipeline_for_dataset(dataset_name: str) -> Optional[Pipeline]:
    """
    Завантажує згенерований ADK MLE-STAR pipeline для датасету.
    """
    # Маппінг датасетів на файли ADK pipelines
    adk_files = {
        "iris": "iris_gbdt_run1_init_code_1.py",
        "california_housing": "california-housing-prices_run1_init_code_1.py",
    }
    
    if dataset_name not in adk_files:
        print(f"  ⚠️ ADK pipeline для {dataset_name} не знайдено")
        return None
    
    filepath = GENERATED_PIPELINES_DIR / adk_files[dataset_name]
    
    if not filepath.exists():
        print(f"  ⚠️ Файл {filepath} не існує")
        return None
    
    code = filepath.read_text(encoding='utf-8')
    
    # Витягуємо модель з ADK скрипту
    model = extract_model_from_adk_script(code)
    
    if model is None:
        # Fallback: створюємо pipeline на основі типу моделі з коду
        if 'HistGradientBoostingClassifier' in code:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(random_state=42)
        elif 'LGBMRegressor' in code or 'lightgbm' in code:
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    objective='rmse',
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
        elif 'RandomForestClassifier' in code:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
        else:
            print(f"  ⚠️ Не вдалося визначити модель з ADK коду")
            return None
    
    return build_pipeline_from_model(model)


# ============================================================================
# ЕКСПЕРИМЕНТИ
# ============================================================================

def run_prompt_experiment(
    dataset_name: str,
    prompt_type: str,  # "simple" або "mle_star"
    llm_type: str,     # "gemini" або "gpt4o"
    n_runs: int = 5,
) -> Dict[str, Any]:
    """Запускає експеримент з промптом."""
    print(f"\n{'='*60}")
    print(f"Датасет: {dataset_name} | Промпт: {prompt_type} | LLM: {llm_type}")
    print(f"{'='*60}")
    
    info = get_dataset_info(dataset_name)
    
    if prompt_type == "simple":
        template = SIMPLE_PROMPT_TEMPLATE
    else:
        template = MLE_STAR_PROMPT_TEMPLATE
    
    prompt = format_prompt(template, info)
    
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
            "generation_method": f"{prompt_type}_{llm_type}",
            "status": "api_error",
            "error": str(e),
        }
    
    code = extract_code(response)
    
    print("🔧 Будую pipeline...")
    pipeline = execute_code_and_get_pipeline(code)
    
    if pipeline is None:
        return {
            "dataset": dataset_name,
            "prompt_type": prompt_type,
            "llm_type": llm_type,
            "generation_method": f"{prompt_type}_{llm_type}",
            "status": "code_error",
            "code": code[:500],
        }
    
    print(f"📊 Оцінюю pipeline ({n_runs} запусків)...")
    metrics = evaluate_pipeline(
        pipeline, 
        info["X"], 
        info["y"], 
        info["is_classification"],
        n_runs=n_runs,
    )
    
    print(f"✅ Результат: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
    
    pipeline_info = analyze_pipeline_complexity(pipeline)
    
    return {
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "llm_type": llm_type,
        "generation_method": f"{prompt_type}_{llm_type}",
        "status": "success",
        "score_mean": metrics["mean"],
        "score_std": metrics["std"],
        "n_runs": metrics["n_runs"],
        "api_time_sec": api_time,
        **pipeline_info,
    }


def run_adk_experiment(
    dataset_name: str,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """Запускає експеримент з ADK MLE-STAR pipeline."""
    print(f"\n{'='*60}")
    print(f"Датасет: {dataset_name} | Метод: ADK MLE-STAR Agent")
    print(f"{'='*60}")
    
    info = get_dataset_info(dataset_name)
    
    print("🔧 Завантажую ADK pipeline...")
    pipeline = get_adk_pipeline_for_dataset(dataset_name)
    
    if pipeline is None:
        return {
            "dataset": dataset_name,
            "prompt_type": "adk_agent",
            "llm_type": "gemini_adk",
            "generation_method": "adk_mle_star",
            "status": "not_available",
        }
    
    print(f"📊 Оцінюю pipeline ({n_runs} запусків)...")
    metrics = evaluate_pipeline(
        pipeline, 
        info["X"], 
        info["y"], 
        info["is_classification"],
        n_runs=n_runs,
    )
    
    print(f"✅ Результат: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
    
    pipeline_info = analyze_pipeline_complexity(pipeline)
    
    return {
        "dataset": dataset_name,
        "prompt_type": "adk_agent",
        "llm_type": "gemini_adk",
        "generation_method": "adk_mle_star",
        "status": "success",
        "score_mean": metrics["mean"],
        "score_std": metrics["std"],
        "n_runs": metrics["n_runs"],
        "api_time_sec": 0,  # Pre-generated
        **pipeline_info,
    }


def run_full_experiment():
    """Запускає повний комплексний експеримент."""
    print("\n" + "="*70)
    print("🧪 КОМПЛЕКСНИЙ ЕКСПЕРИМЕНТ: ПОРІВНЯННЯ МЕТОДІВ ГЕНЕРАЦІЇ ML PIPELINES")
    print("="*70)
    
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
    
    # 1. Промптові експерименти
    for dataset in datasets:
        for prompt_type in prompt_types:
            for llm_type in llm_types:
                result = run_prompt_experiment(
                    dataset_name=dataset,
                    prompt_type=prompt_type,
                    llm_type=llm_type,
                    n_runs=5,
                )
                results.append(result)
                time.sleep(1)
    
    # 2. ADK MLE-STAR експерименти
    for dataset in datasets:
        result = run_adk_experiment(dataset_name=dataset, n_runs=5)
        results.append(result)
    
    # Зберігаємо результати
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = RESULTS_DIR / f"comprehensive_experiment_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Результати збережено: {csv_path}")
    
    # Виводимо зведену таблицю
    print("\n" + "="*70)
    print("📊 ЗВЕДЕНІ РЕЗУЛЬТАТИ")
    print("="*70)
    
    success_df = df[df["status"] == "success"]
    if len(success_df) > 0:
        summary = success_df.groupby(["dataset", "generation_method"]).agg({
            "score_mean": "mean",
            "score_std": "mean",
            "n_steps": "mean",
            "has_pca": "mean",
            "has_ensemble": "mean",
            "model_type": "first",
        }).round(4)
        print(summary.to_string())
    
    # Порівняння методів
    print("\n" + "="*70)
    print("📈 ПОРІВНЯННЯ МЕТОДІВ ГЕНЕРАЦІЇ")
    print("="*70)
    
    if len(success_df) > 0:
        method_summary = success_df.groupby("generation_method").agg({
            "score_mean": ["mean", "std"],
            "n_steps": "mean",
        }).round(4)
        print(method_summary.to_string())
    
    return df


if __name__ == "__main__":
    run_full_experiment()
