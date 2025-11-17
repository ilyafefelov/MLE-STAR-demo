# src/mle_star_ablation/mle_star_generated_pipeline.py

"""
Цей модуль містить конвеєр (pipeline), який був згенерований агентом MLE-STAR 
з офіційного репозиторію Google ADK (google/adk-samples), та адаптер для 
абляційного аналізу.

АРХІТЕКТУРА:
- build_full_pipeline() -> повертає такий самий Pipeline, як видає MLE-STAR
- build_pipeline(config) -> повертає модифікований Pipeline з вимкненими
  компонентами (scaling, feature engineering, tuning, ensemble)

ДЖЕРЕЛО КОДУ:
Базовий pipeline генерується офіційним MLE-STAR агентом:
https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering

ВИКОРИСТАННЯ:
1. Запустити MLE-STAR агента на датасеті
2. Скопіювати згенерований код у build_full_pipeline()
3. Адаптувати назви кроків у build_pipeline()
4. Запустити абляційний аналіз

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

from copy import deepcopy
from typing import List, Optional
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# ==== 1. ПОВНИЙ PIPELINE ВІД GEMINI API ====================================

__FULL_PIPELINE_FN = None

def set_full_pipeline_callable(fn):
    """
    Register an external build_full_pipeline() function from a Gemini-generated file.
    If not registered, fallback to module's own build_full_pipeline().
    """
    global __FULL_PIPELINE_FN
    __FULL_PIPELINE_FN = fn


def _get_full_pipeline_callable():
    """Return the registered build_full_pipeline or the default one in this module."""
    return __FULL_PIPELINE_FN if __FULL_PIPELINE_FN is not None else build_full_pipeline


def build_full_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Повертає повний ML-конвеєр, згенерований Gemini API для iris.
    
    Pipeline складається з:
    1. Preprocessor: SimpleImputer + StandardScaler
    2. Feature Engineering: PCA (95% variance)
    3. Model: RandomForestClassifier
    
    Згенеровано: Google Gemini 2.5 Flash Lite
    Датасет: iris (150 samples, 4 features, 3 classes)
    Дата: 2025-11-13 13:25:59
    
    Model Choice: RandomForestClassifier with max_depth=5
    - Best balance for small dataset (150 samples)
    - Handles non-linearity without overfitting
    - Robust to outliers
    
    Returns:
        Pipeline: Повний sklearn Pipeline з усіма компонентами
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    
    # Preprocessing: Handle missing values and scale features
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Feature Engineering: Dimensionality reduction.
    # PCA aims to capture the most variance with fewer components.
    # Keeping 95% of variance is a common heuristic. For Iris, it might reduce to 2 or 3 components.
    # An alternative could be SelectKBest with chi2 if we wanted to select the most relevant features directly.
    # PolynomialFeatures could be used for non-linear relationships but might overfit the small Iris dataset.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42))
    ])

    # Model: RandomForestClassifier
    # Chosen due to its robustness, ability to handle non-linearities, and good generalization
    # on small to medium datasets like Iris when hyperparameters are set appropriately.
    # - n_estimators=100: A good balance between performance and computation.
    # - max_depth=5: Controls overfitting on the small dataset. Individual trees are not allowed to grow too deep.
    # - random_state=42: Ensures reproducibility.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Combine all steps into a single pipeline.
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])

    return full_pipeline


# ==== 2. КОНФІГУРАЦІЯ НАЗВ КРОКІВ ==========================================

# Маппінг стандартних назв компонентів на назви кроків у Gemini pipeline
_STEP_NAMES = {
    'preprocessor': 'preprocessor',               # SimpleImputer + StandardScaler
    'feature_engineering': 'feature_engineering', # PCA
    'model': 'model',                             # LogisticRegression
}


# ==== 3. ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ МАНІПУЛЯЦІЇ STEPS ==========================

def _remove_step(pipe: Pipeline, step_name: str) -> Pipeline:
    """
    Повертає копію Pipeline без кроку з даною назвою.
    
    Args:
        pipe: Оригінальний Pipeline
        step_name: Назва кроку для видалення
        
    Returns:
        Pipeline: Новий Pipeline без вказаного кроку
        
    Example:
        >>> pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
        >>> new_pipe = _remove_step(pipe, 'scaler')
        >>> new_pipe.steps
        [('model', LogisticRegression())]
    """
    steps = [(name, est) for name, est in pipe.steps if name != step_name]
    return Pipeline(steps=steps)


def _replace_step(pipe: Pipeline, step_name: str, new_estimator) -> Pipeline:
    """
    Замінює один крок іншим estimator.
    
    Args:
        pipe: Оригінальний Pipeline
        step_name: Назва кроку для заміни
        new_estimator: Новий estimator (або None для видалення)
        
    Returns:
        Pipeline: Новий Pipeline з заміненим кроком
        
    Example:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
        >>> new_pipe = _replace_step(pipe, 'scaler', MinMaxScaler())
    """
    new_steps: List = []
    for name, est in pipe.steps:
        if name == step_name:
            if new_estimator is not None:
                new_steps.append((name, new_estimator))
            # Якщо new_estimator is None, просто пропускаємо крок
        else:
            new_steps.append((name, est))
    return Pipeline(new_steps)


def _has_step(pipe: Pipeline, step_name: str) -> bool:
    """
    Перевіряє, чи містить Pipeline крок з даною назвою.
    
    Args:
        pipe: Pipeline для перевірки
        step_name: Назва кроку
        
    Returns:
        bool: True якщо крок присутній
    """
    return any(name == step_name for name, _ in pipe.steps)


# ==== 4. АДАПТЕР ДЛЯ АБЛЯЦІЙ ==============================================

def build_pipeline(config, **kwargs) -> Pipeline:
    """
    Створює конвеєр на основі Gemini-згенерованого pipeline та конфігурації ablation.
    
    Ця функція бере повний pipeline від Gemini і модифікує його відповідно до
    конфігурації абляції: вимикає preprocessing, інженерію ознак, тощо.
    
    Args:
        config: AblationConfig з параметрами (use_scaling, use_feature_engineering, ...)
        **kwargs: Додаткові аргументи для build_full_pipeline()
        
    Returns:
        Pipeline: Модифікований sklearn Pipeline
        
    Приклад:
        >>> from .config import AblationConfig
        >>> config = AblationConfig(name="no_scaling", use_scaling=False)
        >>> pipe = build_pipeline(config)
        # Pipeline буде без кроку 'preprocessor'
    """
    # Отримуємо базовий pipeline від Gemini
    base = _get_full_pipeline_callable()(**kwargs)
    pipe = deepcopy(base)  # Копія, щоб не модифікувати оригінал
    
    # Preprocessing (включає imputer + scaler)
    # use_scaling контролює весь preprocessing блок
    if not config.use_scaling:
        preprocessor_name = _STEP_NAMES['preprocessor']
        if _has_step(pipe, preprocessor_name):
            pipe = _remove_step(pipe, preprocessor_name)
    
    # Інженерія ознак (PCA в нашому випадку)
    if not config.use_feature_engineering:
        fe_name = _STEP_NAMES['feature_engineering']
        if _has_step(pipe, fe_name):
            pipe = _remove_step(pipe, fe_name)
    
    return pipe


# ==== 5. ІНСПЕКЦІЯ PIPELINE ================================================

def inspect_pipeline(pipe: Pipeline) -> dict:
    """
    Повертає структуровану інформацію про Pipeline для аналізу.
    
    Args:
        pipe: Pipeline для інспекції
        
    Returns:
        dict: Інформація про кроки pipeline
        
    Example:
        >>> pipe = build_full_pipeline()
        >>> info = inspect_pipeline(pipe)
        >>> print(info['steps'])
        ['scaler', 'feature_engineering', 'feature_selection', 'model']
    """
    return {
        'steps': [name for name, _ in pipe.steps],
        'n_steps': len(pipe.steps),
        'has_scaler': _has_step(pipe, _STEP_NAMES['scaler']),
        'has_feature_engineering': _has_step(pipe, _STEP_NAMES['feature_engineering']),
        'has_feature_selection': _has_step(pipe, _STEP_NAMES.get('feature_selection', 'feature_selection')),
        'model_type': type(pipe.steps[-1][1]).__name__ if pipe.steps else None,
    }


def print_pipeline_structure(pipe: Pipeline) -> None:
    """
    Друкує структуру Pipeline у читабельному форматі.
    
    Args:
        pipe: Pipeline для відображення
    """
    print("\n" + "="*60)
    print("MLE-STAR PIPELINE STRUCTURE")
    print("="*60)
    
    for i, (name, estimator) in enumerate(pipe.steps, 1):
        print(f"\n{i}. {name}")
        print(f"   Type: {type(estimator).__name__}")
        
        # Додаткова інформація для ColumnTransformer
        if isinstance(estimator, ColumnTransformer):
            print(f"   Transformers: {len(estimator.transformers)}")
            for trans_name, trans, cols in estimator.transformers:
                print(f"      - {trans_name}: {type(trans).__name__}")
    
    print("\n" + "="*60)


# ==== 6. ЕКСПОРТ ===========================================================

__all__ = [
    'build_full_pipeline',
    'build_pipeline',
    'inspect_pipeline',
    'print_pipeline_structure',
]
