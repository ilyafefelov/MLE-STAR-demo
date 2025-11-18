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
from typing import List, Optional, Any
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier


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


def _ensure_pipeline_object(obj: Any) -> Pipeline:
    """
    Ensure the returned object from a builder is a Pipeline-like estimator.
    Accepts: Pipeline, tuple (first element is estimator), dict with 'pipeline' key, or GridSearchCV wrapper.
    Returns the fitted or unfitted estimator (sklearn Pipeline-like).
    """
    from sklearn.pipeline import Pipeline as SKPipeline
    # If the object is already a pipeline/estimator with fit method
    if hasattr(obj, 'fit'):
        return obj
    # Tuple/list: find first element with fit
    if isinstance(obj, (tuple, list)):
        for el in obj:
            if hasattr(el, 'fit'):
                return el
        # fallback: try first element
        return obj[0]
    # dict with 'pipeline' key
    if isinstance(obj, dict) and 'pipeline' in obj:
        p = obj['pipeline']
        if hasattr(p, 'fit'):
            return p
    # Not recognized - raise ValueError
    raise ValueError('Returned object from builder is not a pipeline or estimator')


def build_full_pipeline(
    random_state: int = 42,
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
        ('pca', PCA(n_components=0.95, random_state=random_state))
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
        random_state=random_state
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


def _unwrap_hyperparam_tuner(estimator):
    """If estimator is a hyperparam search wrapper, return the inner estimator template.
    This doesn't require the search to be fitted; we return the `estimator` attribute.
    """
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
        return estimator.estimator
    return estimator


def _unwrap_ensemble(estimator):
    """If estimator is an ensemble (VotingClassifier/Stacking), pick an appropriate
    single estimator fallback so the ablation can run without ensemble.
    """
    if isinstance(estimator, VotingClassifier):
        # Use the first estimator in the list
        if estimator.estimators and len(estimator.estimators) > 0:
            return estimator.estimators[0][1]
    if isinstance(estimator, StackingClassifier):
        # If stacking, return the final_estimator or the first base estimator as fallback
        if hasattr(estimator, 'final_estimator') and estimator.final_estimator is not None:
            return estimator.final_estimator
        elif estimator.estimators and len(estimator.estimators) > 0:
            return estimator.estimators[0][1]
    if isinstance(estimator, BaggingClassifier):
        # For bagging, use the base estimator
        return estimator.base_estimator
    return estimator


def _set_n_jobs_on_estimator(estimator, n_jobs=1):
    """Traverse an estimator or pipeline and set n_jobs=1 where applicable.
    This is a best-effort enforcement for determinism.
    """
    # If a pipeline, set for nested steps
    if isinstance(estimator, Pipeline):
        for _, nested in estimator.steps:
            _set_n_jobs_on_estimator(nested, n_jobs=n_jobs)
        return
    # ColumnTransformer
    if isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers:
            _set_n_jobs_on_estimator(transformer, n_jobs=n_jobs)
        return
    # If the estimator supports set_params and has n_jobs param
    try:
        params = estimator.get_params()
        if 'n_jobs' in params:
            try:
                estimator.set_params(n_jobs=n_jobs)
            except Exception:
                pass
    except Exception:
        pass


def _set_random_state_on_estimator(estimator, random_state=42):
    """Traverse an estimator or pipeline and set random_state where applicable.
    This is a best-effort enforcement for determinism.
    """
    # If a pipeline, set for nested steps
    if isinstance(estimator, Pipeline):
        for _, nested in estimator.steps:
            _set_random_state_on_estimator(nested, random_state=random_state)
        return
    # ColumnTransformer wrap nested transformers
    if isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers:
            _set_random_state_on_estimator(transformer, random_state=random_state)
        return
    # If an ensemble with base estimator(s)
    try:
        params = estimator.get_params()
        # If direct random_state argument is in params, try to set it
        if 'random_state' in params:
            try:
                estimator.set_params(random_state=random_state)
            except Exception:
                pass
        # Also try to set on known nested estimator names
        for nested_name in ('estimator', 'base_estimator', 'final_estimator'):
            if nested_name in params:
                nested = params.get(nested_name)
                if nested is not None:
                    _set_random_state_on_estimator(nested, random_state=random_state)
    except Exception:
        pass


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


def _step_contains_estimator(pipe: Pipeline, estimator_cls) -> bool:
    """
    Check if any step in the pipeline (or nested pipeline) contains an estimator of type estimator_cls.
    """
    for name, est in pipe.steps:
        # Direct check
        if isinstance(est, estimator_cls):
            return True
        # If it's a Pipeline, check nested steps
        if isinstance(est, Pipeline):
            for _, nested in est.steps:
                if isinstance(nested, estimator_cls):
                    return True
        # ColumnTransformer -> check transformers
        if isinstance(est, ColumnTransformer):
            for _, transformer, _ in est.transformers:
                # transformer can be Pipeline or estimator
                if isinstance(transformer, estimator_cls):
                    return True
                if isinstance(transformer, Pipeline):
                    for _, nested in transformer.steps:
                        if isinstance(nested, estimator_cls):
                            return True
    return False


# ==== 4. АДАПТЕР ДЛЯ АБЛЯЦІЙ ==============================================

def build_pipeline(config, deterministic: bool = False, **kwargs) -> Pipeline:
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
    import tempfile, os
    # Run builder in a temporary directory to avoid local train.csv influencing builder behavior
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                base = _get_full_pipeline_callable()(**kwargs)
            except TypeError:
                # Fallback for older generated build_full_pipeline() that doesn't accept kwargs
                base = _get_full_pipeline_callable()()
            finally:
                os.chdir(old_cwd)
    except TypeError:
        # One more fallback - in case tempdir context raised earlier
        base = _get_full_pipeline_callable()(**kwargs)
    # Unwrap tuple/dict/global returns to ensure we have a Pipeline/estimator
    try:
        base = _ensure_pipeline_object(base)
    except Exception as e:
        warnings.warn(f"Could not extract pipeline object from builder: {e}")
        # Keep original - may raise later
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

    # Тюнінг гіперпараметрів (GridSearchCV, RandomizedSearchCV) - розгортаємо
    if not config.use_hyperparam_tuning:
        # If 'model' step is present, unwrap GridSearch/Randomized
        model_name = _STEP_NAMES['model']
        if _has_step(pipe, model_name):
            # find step and replace with its inner estimator template
            for name, est in pipe.steps:
                if name == model_name:
                    new_est = _unwrap_hyperparam_tuner(est)
                    pipe = _replace_step(pipe, model_name, new_est)
                    break

    # Ансамблювання (Voting/Stacking/etc.) - розгортаємо до окремої моделі
    if not config.use_ensembling:
        model_name = _STEP_NAMES['model']
        if _has_step(pipe, model_name):
            for name, est in pipe.steps:
                if name == model_name:
                    new_est = _unwrap_ensemble(est)
                    pipe = _replace_step(pipe, model_name, new_est)
                    break

    # Якщо запитували детермінізм - намагаємось виставити n_jobs=1 скрізь
    if deterministic:
        _set_n_jobs_on_estimator(pipe, n_jobs=1)
        # If the caller passed random_state in kwargs, respect it; else fallback to 42
        rs = kwargs.get('random_state', 42)
        _set_random_state_on_estimator(pipe, random_state=rs)
    
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier

    has_scaler = _step_contains_estimator(pipe, StandardScaler)
    has_feature_engineering = _has_step(pipe, _STEP_NAMES['feature_engineering'])
    has_feature_selection = _has_step(pipe, _STEP_NAMES.get('feature_selection', 'feature_selection'))
    has_tuning = _step_contains_estimator(pipe, GridSearchCV) or _step_contains_estimator(pipe, RandomizedSearchCV)
    has_ensembling = _step_contains_estimator(pipe, VotingClassifier) or _step_contains_estimator(pipe, StackingClassifier) or _step_contains_estimator(pipe, BaggingClassifier)

    return {
        'steps': [name for name, _ in pipe.steps],
        'n_steps': len(pipe.steps),
        'has_scaler': has_scaler,
        'has_feature_engineering': has_feature_engineering,
        'has_feature_selection': has_feature_selection,
        'has_tuning': has_tuning,
        'has_ensembling': has_ensembling,
        'model_type': type(pipe.steps[-1][1]).__name__ if pipe.steps else None,
    }


def is_ablation_meaningful(pipe: Pipeline, required_min_components: int = 2) -> bool:
    """Simple heuristic to decide whether ablation will be meaningful.
    If the pipeline contains at least `required_min_components` of (scaler, feature_engineering, tuning, ensembling)
    we return True.
    """
    info = inspect_pipeline(pipe)
    comps = [info['has_scaler'], info['has_feature_engineering'], info['has_tuning'], info['has_ensembling']]
    return sum(bool(x) for x in comps) >= required_min_components


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
