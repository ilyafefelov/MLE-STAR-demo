#!/usr/bin/env python
"""
Підготовка task файлів для MLE-STAR з sklearn датасетів.
Створює структуру:
  tasks/
    breast_cancer/
      task_description.txt
      train.csv
      test.csv
    wine/...
    digits/...
    iris/...

Автор: Фефелов Ілля Олександрович
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Додаємо src до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mle_star_ablation.datasets import DatasetLoader


def create_task_files(dataset_name: str, output_base: Path):
    """
    Створює task файли для одного датасету.
    
    Args:
        dataset_name: Назва датасету (breast_cancer, wine, digits, iris)
        output_base: Базова директорія для збереження
    """
    # Завантаження датасету
    print(f"\n📦 Processing {dataset_name}...")
    X_train, X_test, y_train, y_test = DatasetLoader.load_dataset(dataset_name)
    info = DatasetLoader.get_dataset_info(dataset_name)
    
    # Отримання feature names зі sklearn датасету
    loader = DatasetLoader.BUILTIN_DATASETS[dataset_name]
    dataset = loader()
    feature_names = list(dataset.feature_names)
    
    # Створення директорії
    task_dir = output_base / dataset_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Формування DataFrame (вже маємо train/test split)
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    # test.csv НЕ містить target (як у Kaggle)
    
    # Збереження CSV
    train_csv = task_dir / "train.csv"
    test_csv = task_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"  ✅ Created {train_csv} ({len(train_df)} samples)")
    print(f"  ✅ Created {test_csv} ({len(test_df)} samples)")
    
    # Опис датасету
    dataset_descriptions = {
        'breast_cancer': 'Breast Cancer Wisconsin (Diagnostic) - predict if tumor is benign or malignant',
        'wine': 'Wine Recognition - classify wines from different cultivars',
        'digits': 'Optical Recognition of Handwritten Digits - classify handwritten digits 0-9',
        'iris': 'Iris Flower Classification - classify iris species based on sepal/petal measurements'
    }
    
    # Опис задачі
    task_description = f"""# Task

Predict the target class for the {dataset_name} dataset.

This is a classification problem with {info['n_classes']} classes.

Dataset: {dataset_descriptions.get(dataset_name, dataset_name)}

# Metric

accuracy

Note: The model should predict class labels (integers from 0 to {info['n_classes']-1}).

# Submission Format
```
target
1
0
2
etc.
```

# Dataset

train.csv
```
{','.join(feature_names)},target
{','.join(map(str, X_train[0]))},{y_train[0]}
{','.join(map(str, X_train[1]))},{y_train[1]}
{','.join(map(str, X_train[2]))},{y_train[2]}
etc.
```

test.csv
```
{','.join(feature_names)}
{','.join(map(str, X_test[0]))}
{','.join(map(str, X_test[1]))}
{','.join(map(str, X_test[2]))}
etc.
```

# Additional Information

- Number of samples: {info['n_samples']} ({len(train_df)} train, {len(test_df)} test)
- Number of features: {info['n_features']}
- Number of classes: {info['n_classes']}
- Feature types: All features are numeric (real-valued)

# Objective

Build a scikit-learn pipeline that:
1. Preprocesses the data (handle missing values, scaling, etc.)
2. Optionally performs feature engineering
3. Trains a classification model
4. Achieves high accuracy on the test set

The pipeline should be robust and follow machine learning best practices.
"""
    
    task_desc_file = task_dir / "task_description.txt"
    with open(task_desc_file, 'w', encoding='utf-8') as f:
        f.write(task_description)
    
    print(f"  ✅ Created {task_desc_file}")
    
    # Також зберігаємо test labels окремо (для evaluation)
    test_labels_file = task_dir / "test_labels.csv"
    pd.DataFrame({'target': y_test}).to_csv(test_labels_file, index=False)
    print(f"  ✅ Created {test_labels_file} (for evaluation only)")


def main():
    # Визначення шляхів
    mle_star_root = Path(__file__).parent.parent / "adk-samples" / "python" / "agents" / "machine-learning-engineering"
    tasks_dir = mle_star_root / "machine_learning_engineering" / "tasks"
    
    if not tasks_dir.exists():
        print(f"❌ MLE-STAR tasks directory not found: {tasks_dir}")
        print("   Make sure adk-samples is cloned in the project root")
        return
    
    print("="*80)
    print("PREPARING MLE-STAR TASKS FROM SKLEARN DATASETS")
    print("="*80)
    print(f"Output directory: {tasks_dir}")
    
    # Датасети
    datasets = ['breast_cancer', 'wine', 'digits', 'iris']
    
    for dataset_name in datasets:
        try:
            create_task_files(dataset_name, tasks_dir)
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ TASK PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review task files in:")
    print(f"   {tasks_dir}")
    print("2. Run MLE-STAR for each dataset:")
    print("   cd adk-samples/python/agents/machine-learning-engineering")
    print("   poetry run adk run machine_learning_engineering --task breast_cancer")
    print("="*80)


if __name__ == "__main__":
    main()
