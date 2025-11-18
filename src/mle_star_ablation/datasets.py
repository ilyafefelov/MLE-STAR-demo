"""
Модуль для завантаження та підготовки датасетів.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import (
    load_breast_cancer, 
    load_wine, 
    load_digits, 
    load_iris
)
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Клас для завантаження та підготовки датасетів."""
    
    BUILTIN_DATASETS = {
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
        'digits': load_digits,
        'iris': load_iris
    }
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Завантажує датасет та розбиває на train/test.
        
        Args:
            dataset_name: Назва вбудованого датасету або 'custom' для CSV
            csv_path: Шлях до CSV файлу (якщо dataset_name='custom')
            target_column: Назва колонки з цільовою змінною (для CSV)
            test_size: Розмір тестової вибірки
            random_state: Seed для відтворюваності
            
        Returns:
            Tuple[X_train, X_test, y_train, y_test]
        """
        if dataset_name in DatasetLoader.BUILTIN_DATASETS:
            X, y = DatasetLoader._load_builtin(dataset_name)
        elif dataset_name == 'custom' and csv_path is not None:
            X, y = DatasetLoader._load_csv(csv_path, target_column)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DatasetLoader.BUILTIN_DATASETS.keys())} or 'custom'"
            )
        
        # Базовий препроцесинг
        X, y = DatasetLoader._preprocess(X, y)
        
        # Розбивка на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def _load_builtin(name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Завантажує вбудований датасет sklearn."""
        loader = DatasetLoader.BUILTIN_DATASETS[name]
        data = loader()
        # Convert to DataFrame to keep feature names and support ColumnTransformer
        # Avoid ambiguous truth checks on arrays; explicitly handle feature_names
        feature_names = getattr(data, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(data.data.shape[1])]
        df = pd.DataFrame(data=data.data, columns=feature_names)
        return df, data.target
    
    @staticmethod
    def _load_csv(
        csv_path: str, 
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Завантажує датасет з CSV файлу.
        
        Args:
            csv_path: Шлях до CSV
            target_column: Назва колонки з цільовою змінною
            
        Returns:
            Tuple[X, y]
        """
        df = pd.read_csv(csv_path)
        
        if target_column is None:
            # Припускаємо, що остання колонка - це target
            target_column = df.columns[-1]
            print(f"Target column not specified. Using last column: {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        y = df[target_column].values
        X = df.drop(columns=[target_column])
        
        return X, y
    
    @staticmethod
    def _preprocess(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Базовий препроцесинг: видалення пропусків та перевірка даних.
        
        Args:
            X: Матриця ознак
            y: Вектор міток
            
        Returns:
            Tuple[X_clean, y_clean]
        """
        # Перевірка на NaN: use pandas utilities if DataFrame, otherwise numpy
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                mask = ~X.isna().any(axis=1)
            else:
                mask = ~np.isnan(X).any(axis=1)
        except Exception:
            mask = ~np.isnan(X).any(axis=1)
        if mask is not None and (mask.sum() < len(mask)):
            print("Warning: NaN values detected in features. Removing rows with NaN...")
            X = X[mask]
            y = y[mask]
        
        if np.isnan(y).any():
            print("Warning: NaN values detected in target. Removing rows with NaN...")
            mask_y = ~np.isnan(y)
            X = X[mask_y]
            y = y[mask_y]
        
        # Перевірка на inf
        # Replace infinite values if any
        try:
            if np.isinf(np.asarray(X)).any():
                print("Warning: Inf values detected in features. Replacing with max finite value...")
                X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        except Exception:
            pass
        
        return X, y
    
    @staticmethod
    def get_dataset_info(
        dataset_name: str, 
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> dict:
        """
        Повертає інформацію про датасет.
        
        Args:
            dataset_name: Назва датасету
            csv_path: Шлях до CSV (опціонально)
            target_column: Назва цільової колонки (опціонально)
            
        Returns:
            dict: Інформація про датасет
        """
        X_train, X_test, y_train, y_test = DatasetLoader.load_dataset(
            dataset_name, csv_path, target_column
        )
        
        info = {
            'name': dataset_name,
            'n_samples': len(X_train) + len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': {
                int(label): int(count) 
                for label, count in zip(*np.unique(y_train, return_counts=True))
            }
        }
        
        return info


def list_available_datasets() -> list[str]:
    """Повертає список доступних вбудованих датасетів."""
    return list(DatasetLoader.BUILTIN_DATASETS.keys())


def load_dataset(
    dataset_name: str,
    csv_path: Optional[str] = None,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Обгортка над DatasetLoader.load_dataset для зручності імпорту.
    
    Args:
        dataset_name: Назва вбудованого датасету або 'custom' для CSV
        csv_path: Шлях до CSV файлу (якщо dataset_name='custom')
        target_column: Назва колонки з цільовою змінною (для CSV)
        test_size: Розмір тестової вибірки
        random_state: Seed для відтворюваності
        
    Returns:
        Tuple[X_train, X_test, y_train, y_test]
    """
    return DatasetLoader.load_dataset(
        dataset_name, csv_path, target_column, test_size, random_state
    )


def print_dataset_info(dataset_name: str):
    """Виводить інформацію про датасет."""
    info = DatasetLoader.get_dataset_info(dataset_name)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {info['name']}")
    print(f"{'='*50}")
    print(f"Total samples: {info['n_samples']}")
    print(f"Features: {info['n_features']}")
    print(f"Classes: {info['n_classes']}")
    print(f"Train/Test split: {info['train_size']}/{info['test_size']}")
    print(f"\nClass distribution (train):")
    for label, count in info['class_distribution'].items():
        print(f"  Class {label}: {count} samples ({count/info['train_size']*100:.1f}%)")
    print(f"{'='*50}\n")
