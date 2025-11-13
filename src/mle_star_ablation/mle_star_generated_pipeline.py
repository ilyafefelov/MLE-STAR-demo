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

def build_full_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Повертає повний ML-конвеєр, згенерований Gemini API для iris.
    
    Pipeline складається з:
    1. Preprocessor: SimpleImputer + StandardScaler
    2. Feature Engineering: (визначено Gemini)
    3. Model: (визначено Gemini)
    
    Згенеровано: Google Gemini 2.0 Flash Exp
    Датасет: iris
    Дата: 2025-11-13 13:25:59
    
    Returns:
        Pipeline: Повний sklearn Pipeline з усіма компонентами
    """
"""
    Builds a complete scikit-learn ML pipeline for the Iris dataset.

    The pipeline includes preprocessing, feature engineering, and a classification model.

    Dataset Characteristics:
    - Samples: 150
    - Features: 4
    - Classes: 3
    - Task: Classification

    Model Selection Rationale:
    The Iris dataset is a classic benchmark dataset known for its relatively clean data and
    linearly separable nature (especially between classes 0 and 1, and 1 and 2).
    However, the boundary between class 0 and class 2 can be slightly non-linear.
    Given the small dataset size (150 samples) and low feature count (4), simpler models
    can often perform very well without overfitting.

    - LogisticRegression: While good for linearly separable data, it might struggle
      slightly with the mild non-linearity between some classes.
    - SVC with RBF kernel: Excellent for complex decision boundaries and can handle
      non-linearity well. However, with a small dataset and relatively simple structure,
      it might be prone to overfitting or require careful hyperparameter tuning.
    - MLPClassifier: A powerful model capable of learning complex patterns. For this
      small, low-dimensional dataset, it's likely overkill and highly prone to
      overfitting if not heavily regularized. Training can also be slower.
    - GradientBoostingClassifier: A powerful ensemble method that often achieves high
      accuracy. It can be susceptible to overfitting on small datasets if not tuned
      properly.

    - RandomForestClassifier: This is chosen as the "BEST" model for this dataset
      because it offers a good balance.
        - Robustness: It's generally robust to outliers and doesn't assume a specific
          data distribution.
        - Non-linearity: It can naturally handle non-linear relationships between
          features and the target variable.
        - Overfitting: With appropriate hyperparameters (like `max_depth`), it can
          effectively control overfitting on smaller datasets.
        - Interpretability (relative): While not as interpretable as a single decision
          tree, the concept of ensemble of trees provides some level of understanding.
        - Dataset Size Influence: For a dataset of 150 samples, a Random Forest with
          controlled depth is less likely to overfit than a highly flexible SVC or an
          MLP. It's more powerful than Logistic Regression for potential mild non-linearities.

    Hyperparameter Selection:
    - n_estimators=100: A common starting point for the number of trees. More trees
      generally lead to better performance but also increase computation. 100 is usually
      sufficient for this dataset to stabilize performance.
    - max_depth=5: This is a crucial hyperparameter for controlling overfitting on small
      datasets. Limiting the depth of individual trees prevents them from learning the
      noise in the data. Given the low dimensionality, trees don't need to be excessively deep.
      A depth of 5 is a reasonable starting point that balances model complexity and
      generalization for Iris.
    - random_state=42: Ensures reproducibility of the model's training process.

    Feature Engineering Choice (PCA):
    - PCA(n_components=0.95): Principal Component Analysis is used for dimensionality
      reduction. Setting `n_components` to 0.95 means we keep enough principal components
      to explain 95% of the variance in the data. With only 4 features, PCA might not
      be strictly necessary for this dataset to achieve good performance, but it's a
      sophisticated technique that can be beneficial if features are highly correlated
      or if we were dealing with more features. For Iris, it can slightly regularize
      and potentially speed up training without significant loss of information.
      An alternative could be to use all 4 components (e.g., `n_components=4`) if
      variance explained wasn't a concern, or even omit PCA if it doesn't improve
      results significantly.

    Expected Performance Characteristics:
    - High accuracy is expected, likely in the 95-99% range on test data.
    - The model should generalize well due to the choice of RandomForest and controlled
      `max_depth`, and the presence of PCA.
    - Training time should be relatively fast given the dataset size.

    Returns:
        sklearn.pipeline.Pipeline: The configured scikit-learn pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import PolynomialFeatures # Example for more complex feature engineering

    # Preprocessing: Handle missing values and scale features.
    # SimpleImputer is used as a robust first step, though Iris typically has no missing values.
    # StandardScaler is essential for distance-based algorithms and gradient-based optimizers.
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
    base = build_full_pipeline(**kwargs)
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
