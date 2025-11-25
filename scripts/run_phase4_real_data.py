#!/usr/bin/env python3
"""
Фаза 4: Тестування на реальних даних (Telecom Churn)
====================================================

Перевірка гіпотези over-engineering на реальному промисловому датасеті.

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_TOKEN")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "real_data_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_telecom_churn():
    """Завантажує та готує Telecom Churn датасет."""
    df = pd.read_csv(DATA_DIR / "telecom_churn.csv")
    
    # Видаляємо CustomerID
    df = df.drop('CustomerID', axis=1)
    
    # Кодуємо категоріальні змінні
    cat_cols = ['Gender', 'InternetService', 'Contract', 'PaymentMethod', 'PaperlessBilling']
    num_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Label encoding для простоти
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    
    return X, y, df.drop('Churn', axis=1).columns.tolist()


def create_simple_pipeline():
    """Простий pipeline (Simple prompt style)."""
    from sklearn.svm import SVC
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC(random_state=42, probability=True))
    ])


def create_mle_star_pipeline():
    """Складний pipeline (MLE-STAR style з over-engineering)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Base estimators
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', ensemble)
    ])


def create_minimal_pipeline():
    """Мінімальний pipeline (тільки модель)."""
    from sklearn.ensemble import GradientBoostingClassifier
    
    return Pipeline([
        ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])


def create_optimal_pipeline():
    """Оптимальний pipeline (на основі наших висновків)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])


def evaluate_pipeline(pipeline, X, y, name, n_runs=5):
    """Оцінює pipeline."""
    scores = []
    f1_scores = []
    
    for run in range(n_runs):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run)
        
        # Accuracy
        acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(np.mean(acc_scores))
        
        # F1
        f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
        f1_scores.append(np.mean(f1))
    
    return {
        'name': name,
        'accuracy_mean': np.mean(scores),
        'accuracy_std': np.std(scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'n_runs': n_runs,
    }


def run_phase4_experiment():
    """Запускає Фазу 4: тестування на реальних даних."""
    print("=" * 70)
    print("📊 ФАЗА 4: ТЕСТУВАННЯ НА РЕАЛЬНИХ ДАНИХ (TELECOM CHURN)")
    print("=" * 70)
    
    # Завантажуємо дані
    print("\n📁 Завантажую Telecom Churn датасет...")
    X, y, feature_names = load_telecom_churn()
    print(f"   • Зразків: {X.shape[0]}")
    print(f"   • Ознак: {X.shape[1]}")
    print(f"   • Класи: {np.unique(y)} (Churn: {np.sum(y)}, No Churn: {len(y) - np.sum(y)})")
    
    # Тестуємо різні конфігурації
    pipelines = {
        'Simple (SVC)': create_simple_pipeline(),
        'MLE-STAR (VotingClassifier + PCA)': create_mle_star_pipeline(),
        'Minimal (GradientBoosting only)': create_minimal_pipeline(),
        'Optimal (Scaler + HistGB)': create_optimal_pipeline(),
    }
    
    results = []
    
    print("\n🧪 Тестування конфігурацій...")
    for name, pipeline in pipelines.items():
        print(f"\n   → {name}")
        start = time.time()
        result = evaluate_pipeline(pipeline, X, y, name)
        elapsed = time.time() - start
        result['time_sec'] = elapsed
        results.append(result)
        print(f"     Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"     F1-score: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
        print(f"     Час: {elapsed:.1f}с")
    
    # Зберігаємо результати
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"telecom_churn_experiment_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Виводимо зведену таблицю
    print("\n" + "=" * 70)
    print("📈 РЕЗУЛЬТАТИ ФАЗИ 4")
    print("=" * 70)
    
    df_sorted = df.sort_values('accuracy_mean', ascending=False)
    print("\nРейтинг по Accuracy:")
    for i, row in df_sorted.iterrows():
        print(f"   {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['name']}")
    
    print("\nРейтинг по F1-score:")
    df_f1 = df.sort_values('f1_mean', ascending=False)
    for i, row in df_f1.iterrows():
        print(f"   {row['f1_mean']:.4f} ± {row['f1_std']:.4f} | {row['name']}")
    
    # Аналіз
    best_acc = df_sorted.iloc[0]
    mle_star = df[df['name'].str.contains('MLE-STAR')].iloc[0]
    
    print("\n" + "=" * 70)
    print("📊 АНАЛІЗ")
    print("=" * 70)
    
    if best_acc['name'] != mle_star['name']:
        delta = best_acc['accuracy_mean'] - mle_star['accuracy_mean']
        print(f"\n✅ ГІПОТЕЗА ПІДТВЕРДЖЕНА на реальних даних!")
        print(f"   • Переможець: {best_acc['name']}")
        print(f"   • MLE-STAR: {mle_star['accuracy_mean']:.4f}")
        print(f"   • Різниця: +{delta:.4f} ({delta*100:.2f}%)")
        print(f"\n   Over-engineering (VotingClassifier + PCA) програє на реальних даних!")
    else:
        print(f"\n⚠️ MLE-STAR виявився найкращим на цьому датасеті")
    
    print(f"\n📁 Результати збережено: {csv_path}")
    
    return df


if __name__ == "__main__":
    run_phase4_experiment()
