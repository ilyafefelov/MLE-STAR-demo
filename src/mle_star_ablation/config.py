# src/mle_star_ablation/config.py

"""
Конфігурація для абляційного аналізу MLE-STAR pipeline.

Цей модуль визначає структуру конфігурацій для систематичного вимикання
компонентів ML-конвеєра та збору метрик їхнього впливу.

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AblationConfig:
    """
    Конфігурація для абляційного експерименту.
    
    Кожна конфігурація визначає, які компоненти pipeline мають бути активними,
    а які вимкнені для оцінки їхнього впливу на результати.
    
    Attributes:
        name: Унікальна назва конфігурації (наприклад, "full", "no_scaling")
        use_scaling: Чи використовувати масштабування ознак (StandardScaler, MinMaxScaler)
        use_feature_engineering: Чи використовувати інженерію ознак (PolynomialFeatures, interactions)
        use_hyperparam_tuning: Чи використовувати підбір гіперпараметрів (GridSearchCV, RandomizedSearchCV)
        use_ensembling: Чи використовувати ансамблювання (VotingClassifier, StackingClassifier)
        model_type: Тип моделі (default, simplified, alternative)
        scaler_type: Тип scaler'а (standard, minmax, robust, none)
        description: Опис конфігурації для звітів
        extra_params: Додаткові параметри (для кастомізації)
        
    Example:
        >>> config_full = AblationConfig(name="full")
        >>> config_no_scaling = AblationConfig(name="no_scaling", use_scaling=False)
        >>> config_minimal = AblationConfig(
        ...     name="minimal",
        ...     use_scaling=False,
        ...     use_feature_engineering=False,
        ...     use_ensembling=False
        ... )
    """
    name: str
    use_scaling: bool = True
    use_feature_engineering: bool = True
    use_hyperparam_tuning: bool = True
    use_ensembling: bool = True
    model_type: str = "default"
    scaler_type: str = "standard"
    description: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Валідація та автоматичне заповнення полів."""
        if self.description is None:
            self.description = self._generate_description()
        
        # Валідація scaler_type
        valid_scalers = {"standard", "minmax", "robust", "none"}
        if self.scaler_type not in valid_scalers:
            raise ValueError(
                f"Invalid scaler_type '{self.scaler_type}'. "
                f"Must be one of {valid_scalers}"
            )
        
        # Якщо use_scaling=False, scaler_type має бути 'none'
        if not self.use_scaling and self.scaler_type != "none":
            self.scaler_type = "none"
    
    def _generate_description(self) -> str:
        """Генерує опис конфігурації на основі активних компонентів."""
        components = []
        if self.use_scaling:
            components.append(f"масштабування ({self.scaler_type})")
        if self.use_feature_engineering:
            components.append("інженерія ознак")
        if self.use_hyperparam_tuning:
            components.append("тюнінг гіперпараметрів")
        if self.use_ensembling:
            components.append("ансамблювання")
        
        if not components:
            return "Мінімальний pipeline (тільки базова модель)"
        
        return f"Pipeline з: {', '.join(components)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертує конфігурацію у словник для збереження у CSV/JSON.
        
        Returns:
            dict: Словник з усіма параметрами конфігурації
        """
        return {
            'name': self.name,
            'use_scaling': self.use_scaling,
            'use_feature_engineering': self.use_feature_engineering,
            'use_hyperparam_tuning': self.use_hyperparam_tuning,
            'use_ensembling': self.use_ensembling,
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'description': self.description,
            **self.extra_params
        }
    
    def get_name(self) -> str:
        """Повертає назву конфігурації (алізація для name)."""
        return self.name
    
    def count_active_components(self) -> int:
        """
        Рахує кількість активних компонентів.
        
        Returns:
            int: Кількість компонентів, які увімкнені
        """
        return sum([
            self.use_scaling,
            self.use_feature_engineering,
            self.use_hyperparam_tuning,
            self.use_ensembling,
        ])
    
    def is_full_pipeline(self) -> bool:
        """Перевіряє, чи це повний pipeline (усі компоненти увімкнені)."""
        return all([
            self.use_scaling,
            self.use_feature_engineering,
            self.use_hyperparam_tuning,
            self.use_ensembling,
        ])
    
    def __repr__(self) -> str:
        """Зручне відображення конфігурації."""
        active = self.count_active_components()
        return f"AblationConfig(name='{self.name}', active_components={active}/4)"


# ==== СТАНДАРТНІ КОНФІГУРАЦІЇ ==============================================

def get_standard_configs() -> list[AblationConfig]:
    """
    Повертає список стандартних конфігурацій для абляційного аналізу.
    
    Ці конфігурації покривають основні сценарії:
    1. Повний pipeline (baseline)
    2. Вимкнення кожного компонента окремо (one-at-a-time ablation)
    3. Мінімальний pipeline (тільки базова модель)
    
    Returns:
        list[AblationConfig]: Список з 6 стандартних конфігурацій
        
    Example:
        >>> configs = get_standard_configs()
        >>> for cfg in configs:
        ...     print(f"{cfg.name}: {cfg.count_active_components()}/4 компонентів")
        full: 4/4 компонентів
        no_scaling: 3/4 компонентів
        ...
    """
    return [
        # 1. Baseline: повний pipeline
        AblationConfig(
            name="full",
            description="Повний pipeline з усіма компонентами (baseline)"
        ),
        
        # 2. Вимикаємо масштабування
        AblationConfig(
            name="no_scaling",
            use_scaling=False,
            description="Pipeline без масштабування ознак"
        ),
        
        # 3. Вимикаємо інженерію ознак
        AblationConfig(
            name="no_feature_engineering",
            use_feature_engineering=False,
            description="Pipeline без інженерії ознак"
        ),
        
        # 4. Вимикаємо тюнінг гіперпараметрів
        AblationConfig(
            name="no_tuning",
            use_hyperparam_tuning=False,
            description="Pipeline без підбору гіперпараметрів"
        ),
        
        # 5. Вимикаємо ансамблювання
        AblationConfig(
            name="no_ensemble",
            use_ensembling=False,
            description="Pipeline без ансамблювання моделей"
        ),
        
        # 6. Мінімальний pipeline
        AblationConfig(
            name="minimal",
            use_scaling=False,
            use_feature_engineering=False,
            use_hyperparam_tuning=False,
            use_ensembling=False,
            description="Мінімальний pipeline (тільки базова модель)"
        ),
    ]


def get_cumulative_configs() -> list[AblationConfig]:
    """
    Повертає конфігурації для кумулятивного аналізу.
    
    Кумулятивний підхід: починаємо з мінімального pipeline і поступово
    додаємо компоненти, оцінюючи приріст від кожного.
    
    Returns:
        list[AblationConfig]: Список з 5 конфігурацій
    """
    return [
        # Рівень 0: тільки модель
        AblationConfig(
            name="stage_0_model_only",
            use_scaling=False,
            use_feature_engineering=False,
            use_hyperparam_tuning=False,
            use_ensembling=False,
        ),
        
        # Рівень 1: модель + scaling
        AblationConfig(
            name="stage_1_with_scaling",
            use_scaling=True,
            use_feature_engineering=False,
            use_hyperparam_tuning=False,
            use_ensembling=False,
        ),
        
        # Рівень 2: модель + scaling + feature engineering
        AblationConfig(
            name="stage_2_with_features",
            use_scaling=True,
            use_feature_engineering=True,
            use_hyperparam_tuning=False,
            use_ensembling=False,
        ),
        
        # Рівень 3: модель + scaling + features + tuning
        AblationConfig(
            name="stage_3_with_tuning",
            use_scaling=True,
            use_feature_engineering=True,
            use_hyperparam_tuning=True,
            use_ensembling=False,
        ),
        
        # Рівень 4: повний pipeline
        AblationConfig(
            name="stage_4_full",
            use_scaling=True,
            use_feature_engineering=True,
            use_hyperparam_tuning=True,
            use_ensembling=True,
        ),
    ]


def create_custom_config(
    name: str,
    **kwargs
) -> AblationConfig:
    """
    Створює кастомну конфігурацію з довільними параметрами.
    
    Args:
        name: Назва конфігурації
        **kwargs: Параметри для AblationConfig
        
    Returns:
        AblationConfig: Нова конфігурація
        
    Example:
        >>> cfg = create_custom_config(
        ...     name="custom_robust",
        ...     use_scaling=True,
        ...     scaler_type="robust",
        ...     use_ensembling=False
        ... )
    """
    return AblationConfig(name=name, **kwargs)


# ==== ЕКСПОРТ ==============================================================

__all__ = [
    'AblationConfig',
    'get_standard_configs',
    'get_cumulative_configs',
    'create_custom_config',
]
