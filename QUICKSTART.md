# Швидкий старт

Цей посібник допоможе вам швидко почати роботу з проєктом абляційного аналізу.

## Встановлення

### 1. Клонуйте репозиторій (або розпакуйте архів)

```bash
cd Diploma
```

### 2. Створіть віртуальне середовище (рекомендовано)

```bash
python -m venv venv
```

**Активація:**
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

### 3. Встановіть залежності

```bash
pip install -r requirements.txt
```

## Перший запуск

### Простий експеримент на вбудованому датасеті

```bash
python scripts/run_ablation.py --dataset breast_cancer --n-runs 3
```

Це запустить абляційний аналіз на датасеті breast_cancer з 3 повторами для кожної конфігурації.

### Результати

Результати будуть збережені в папці `results/`:
- `detailed_results_*.csv` - детальні результати всіх запусків
- `summary_statistics_*.csv` - підсумкова статистика
- `statistical_report_*.txt` - статистичний звіт
- `pairwise_comparisons_*.csv` - попарні порівняння
- `*.png` - графіки

## Приклади використання

### 1. Запуск одиничного експерименту

```bash
python scripts/run_single_experiment.py --dataset breast_cancer --scaling --tuning
```

Параметри:
- `--scaling` - увімкнути масштабування
- `--feature-engineering` - увімкнути інженерію ознак
- `--tuning` - увімкнути тюнінг гіперпараметрів
- `--ensembling` - увімкнути енсамблювання
- `--model {logistic, random_forest, gradient_boosting}` - вибір моделі

### 2. Запуск з власним датасетом

```bash
python scripts/run_ablation.py --dataset custom --csv-path path/to/data.csv --target target_column --n-runs 5
```

### 3. Без створення графіків (швидше)

```bash
python scripts/run_ablation.py --dataset wine --n-runs 3 --no-plots
```

### 4. Детальний вивід

```bash
python scripts/run_ablation.py --dataset iris --n-runs 3 --verbose
```

## Доступні датасети

Вбудовані датасети:
- `breast_cancer` - 569 зразків, 30 ознак, 2 класи (діагностика раку)
- `wine` - 178 зразків, 13 ознак, 3 класи (класифікація вин)
- `digits` - 1797 зразків, 64 ознаки, 10 класів (розпізнавання цифр)
- `iris` - 150 зразків, 4 ознаки, 3 класи (класифікація ірисів)

## Програмне використання

### Приклад Python скрипта

```python
import sys
sys.path.insert(0, 'src')

from mle_star_ablation import (
    AblationConfig,
    build_pipeline,
    DatasetLoader,
    calculate_classification_metrics
)

# Завантаження даних
X_train, X_test, y_train, y_test = DatasetLoader.load_dataset('breast_cancer')

# Створення конфігурації
config = AblationConfig(
    use_scaling=True,
    use_feature_engineering=False,
    use_hyperparam_tuning=True,
    model_type='logistic'
)

# Побудова та тренування пайплайна
pipeline = build_pipeline(config)
pipeline.fit(X_train, y_train)

# Оцінка
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

## Що далі?

1. Перегляньте `notebooks/example_ablation_analysis.ipynb` для детального прикладу
2. Ознайомтесь з модулями в `src/mle_star_ablation/`
3. Налаштуйте власні конфігурації в `configs/ablation_config.yaml`
4. Запустіть повний набір експериментів для вашого диплома

## Допомога

Якщо виникають помилки:

1. Перевірте версію Python (потрібна 3.8+):
   ```bash
   python --version
   ```

2. Переконайтесь, що всі залежності встановлені:
   ```bash
   pip list
   ```

3. Використовуйте `--verbose` для детальної діагностики:
   ```bash
   python scripts/run_ablation.py --dataset breast_cancer --verbose
   ```

## Корисні команди

```bash
# Перегляд структури проєкту
tree /F   # Windows
ls -R     # Linux/Mac

# Перегляд доступних датасетів
python -c "from src.mle_star_ablation import list_available_datasets; print(list_available_datasets())"

# Швидкий тест на маленькому датасеті
python scripts/run_ablation.py --dataset iris --n-runs 2 --no-plots
```
