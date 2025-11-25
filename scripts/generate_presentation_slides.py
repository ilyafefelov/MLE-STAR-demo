#!/usr/bin/env python3
"""
Генератор слайдів для презентації дипломної роботи
===================================================

Створює основні графіки та матеріали для презентації:
1. Порівняння методів генерації (Simple vs MLE-STAR vs ADK)
2. Ablation аналіз (Forest Plot)
3. Ключові метрики та висновки

Автор: Фефелов Ілля Олександрович
МАУП, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Шляхи
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_comparison"
OUTPUT_DIR = REPORTS_DIR / "figures" / "presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Налаштування стилю
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.figsize'] = (12, 8)


def create_llm_comparison_chart():
    """Створює графік порівняння методів генерації LLM."""
    
    # Дані з експерименту
    data = {
        'Датасет': ['Iris', 'Iris', 'Iris', 'Iris', 'Iris',
                   'Breast Cancer', 'Breast Cancer', 'Breast Cancer', 'Breast Cancer',
                   'California Housing', 'California Housing', 'California Housing', 'California Housing', 'California Housing'],
        'Метод': ['Simple Gemini', 'Simple GPT-4o', 'MLE-STAR Gemini', 'MLE-STAR GPT-4o', 'ADK Agent',
                 'Simple Gemini', 'Simple GPT-4o', 'MLE-STAR Gemini', 'MLE-STAR GPT-4o',
                 'Simple Gemini', 'Simple GPT-4o', 'MLE-STAR Gemini', 'MLE-STAR GPT-4o', 'ADK Agent'],
        'Score': [0.963, 0.963, 0.961, 0.904, 0.947,
                 0.975, 0.975, 0.962, 0.965,
                 0.602, 0.602, 0.766, 0.661, 0.837],
    }
    
    df = pd.DataFrame(data)
    
    # Окремо для класифікації та регресії
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Класифікація
    clf_data = df[df['Датасет'].isin(['Iris', 'Breast Cancer'])]
    clf_pivot = clf_data.pivot(index='Датасет', columns='Метод', values='Score')
    
    clf_pivot.plot(kind='bar', ax=axes[0], width=0.8, 
                   color=['#3498db', '#2980b9', '#e74c3c', '#c0392b', '#27ae60'])
    axes[0].set_title('Класифікація: Accuracy', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_xlabel('')
    axes[0].legend(title='Метод генерації', loc='lower right', fontsize=10)
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].axhline(y=0.963, color='green', linestyle='--', alpha=0.5, label='Best Simple')
    
    # Регресія  
    reg_data = df[df['Датасет'] == 'California Housing']
    x = range(len(reg_data))
    bars = axes[1].bar(x, reg_data['Score'], 
                       color=['#3498db', '#2980b9', '#e74c3c', '#c0392b', '#27ae60'])
    axes[1].set_title('Регресія: R² Score', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reg_data['Метод'], rotation=45, ha='right')
    axes[1].set_ylim(0.5, 0.9)
    
    # Позначення переможця
    max_idx = reg_data['Score'].values.argmax()
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "llm_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Збережено: {OUTPUT_DIR / 'llm_comparison.png'}")
    plt.close()


def create_overengineering_evidence():
    """Створює графік доказу over-engineering."""
    
    # Ключові приклади
    data = {
        'Конфігурація': ['Full (Gemini)', 'Minimal', 'no_fe', 
                        'Full (Gemini)', 'Minimal', 'no_fe'],
        'Датасет': ['California Housing', 'California Housing', 'California Housing',
                   'Breast Cancer', 'Breast Cancer', 'Breast Cancer'],
        'Score': [0.78, 0.85, 0.87, 0.970, 0.951, 0.975],
    }
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # California Housing
    cal_data = df[df['Датасет'] == 'California Housing']
    colors = ['#e74c3c', '#27ae60', '#27ae60']
    bars = axes[0].bar(cal_data['Конфігурація'], cal_data['Score'], color=colors)
    axes[0].set_title('California Housing (R²)\nOver-engineering погіршує результат', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0.7, 0.9)
    
    # Стрілка покращення
    axes[0].annotate('', xy=(2, 0.87), xytext=(0, 0.78),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    axes[0].text(1, 0.895, '+11.5%', fontsize=14, ha='center', color='green', fontweight='bold')
    
    # Breast Cancer
    bc_data = df[df['Датасет'] == 'Breast Cancer']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = axes[1].bar(bc_data['Конфігурація'], bc_data['Score'], color=colors)
    axes[1].set_title('Breast Cancer (Accuracy)\nМінімалістичний підхід перемагає', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.94, 0.99)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overengineering_evidence.png", dpi=150, bbox_inches='tight')
    print(f"✅ Збережено: {OUTPUT_DIR / 'overengineering_evidence.png'}")
    plt.close()


def create_method_stability_chart():
    """Створює графік стабільності методів."""
    
    methods = ['ADK Agent', 'MLE-STAR Gemini', 'MLE-STAR GPT-4o', 'Simple Gemini', 'Simple GPT-4o']
    mean_scores = [0.892, 0.897, 0.843, 0.847, 0.847]
    std_scores = [0.08, 0.11, 0.16, 0.21, 0.21]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#27ae60', '#e74c3c', '#c0392b', '#3498db', '#2980b9']
    x = range(len(methods))
    
    bars = ax.bar(x, mean_scores, yerr=std_scores, capsize=10, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Середній Score (across datasets)')
    ax.set_title('Стабільність методів генерації\n(Mean ± Std по всіх датасетах)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0.6, 1.1)
    
    # Позначення найстабільнішого
    ax.annotate('Найстабільніший\n(std=0.08)', xy=(0, 0.95), xytext=(1.5, 1.0),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=12, color='green')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_stability.png", dpi=150, bbox_inches='tight')
    print(f"✅ Збережено: {OUTPUT_DIR / 'method_stability.png'}")
    plt.close()


def create_architecture_comparison():
    """Створює таблицю порівняння архітектур."""
    
    # Дані для таблиці
    col_labels = ['Метод', 'Scaler', 'PCA', 'Ensemble', 'Типова модель', 'Кроків']
    cell_text = [
        ['Simple Prompt', '✓', '✗', '✗', 'SVC / LinearRegression', '2'],
        ['MLE-STAR Prompt', '✓', '✓', '✓', 'VotingClassifier', '3-4'],
        ['ADK Agent', '✓', '✗', '✗', 'HistGradientBoosting / LGBM', '3'],
    ]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(col_labels),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    plt.title('Порівняння архітектур згенерованих пайплайнів', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Збережено: {OUTPUT_DIR / 'architecture_comparison.png'}")
    plt.close()


def create_key_findings_slide():
    """Створює слайд з ключовими висновками."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    findings = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     КЛЮЧОВІ ВИСНОВКИ ДОСЛІДЖЕННЯ                            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  1️⃣  OVER-ENGINEERING ПІДТВЕРДЖЕНО                                           ║
    ║     • LLM додають зайві компоненти (PCA, Ensemble)                          ║
    ║     • Падіння точності до 6% на класифікації                                 ║
    ║     • Cohen's d > 0.8 для California Housing (великий ефект)                ║
    ║                                                                              ║
    ║  2️⃣  ПРОСТОТА ПЕРЕМАГАЄ (для класифікації)                                   ║
    ║     • Simple prompt: 0.963-0.975 Accuracy                                   ║
    ║     • MLE-STAR prompt: 0.904-0.965 Accuracy                                  ║
    ║     • Різниця: до +6% на користь простого підходу                           ║
    ║                                                                              ║
    ║  3️⃣  ADK AGENT ОПТИМАЛЬНИЙ (для регресії)                                    ║
    ║     • ADK: R² = 0.837 (California Housing)                                   ║
    ║     • Simple: R² = 0.602                                                     ║
    ║     • Покращення: +39% R²                                                    ║
    ║                                                                              ║
    ║  4️⃣  ПРАКТИЧНІ РЕКОМЕНДАЦІЇ                                                  ║
    ║     • Класифікація → Simple prompt + SVC                                     ║
    ║     • Регресія на складних даних → ADK MLE-STAR Agent                        ║
    ║     • Scaling критичний, PCA часто зайвий                                    ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, findings, fontsize=11, ha='center', va='center',
            family='monospace', transform=ax.transAxes)
    
    plt.savefig(OUTPUT_DIR / "key_findings.png", dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Збережено: {OUTPUT_DIR / 'key_findings.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("🎨 ГЕНЕРАЦІЯ МАТЕРІАЛІВ ДЛЯ ПРЕЗЕНТАЦІЇ")
    print("=" * 60)
    
    create_llm_comparison_chart()
    create_overengineering_evidence()
    create_method_stability_chart()
    create_architecture_comparison()
    create_key_findings_slide()
    
    print("\n" + "=" * 60)
    print(f"✅ Усі графіки збережено в: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Копіюємо Forest Plot
    forest_src = REPORTS_DIR / "forest_plot_all_datasets.png"
    if forest_src.exists():
        import shutil
        shutil.copy(forest_src, OUTPUT_DIR / "forest_plot.png")
        print(f"✅ Скопійовано Forest Plot")
    
    # Список файлів
    print("\n📁 Згенеровані файли:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()
