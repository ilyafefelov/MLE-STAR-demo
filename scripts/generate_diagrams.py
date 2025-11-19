#!/usr/bin/env python
"""
Generate PNG diagrams for the DATA_FLOW_AND_EXPERIMENT_SETUP doc.

This script creates two diagrams:
 - high_level_flow.png: shows the orchestration flow between scripts and components
 - ablation_examples.png: shows how ablation toggles (no_scaling, no_tuning, no_ensemble) modify pipeline steps

"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'diagrams'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_flowchart(path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    def box(x, y, w, h, text, color='#ffffff'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3', edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
        return rect

    # Boxes
    b1 = box(0.02, 0.7, 0.2, 0.18, 'run_experiment_suite.py\n(manifest)')
    b2 = box(0.28, 0.7, 0.22, 0.18, 'run_ablation.py\n(ablation loops)')
    b3 = box(0.62, 0.86, 0.2, 0.12, 'DatasetLoader\n(load & split)')
    b4 = box(0.62, 0.62, 0.2, 0.12, 'Pipeline Wrapper\n(build_full_pipeline)')
    b5 = box(0.82, 0.86, 0.16, 0.12, 'Adapter\n(build_pipeline)')
    b6 = box(0.82, 0.62, 0.16, 0.12, 'Sklearn Pipeline\n(fit/predict)')
    b7 = box(0.5, 0.34, 0.36, 0.16, 'Metrics + Analysis\n(summary, pairwise, plots)')
    b8 = box(0.5, 0.04, 0.36, 0.16, 'results/<exp>/* (csv, png, txt)')

    # Arrows
    def arrow(a, b):
        ax.annotate('', xy=(a[0], a[1]), xytext=(b[0], b[1]), arrowprops=dict(arrowstyle='->', lw=1.6))

    arrow((0.22, 0.79), (0.28, 0.79))  # suite -> ablation
    arrow((0.5, 0.79), (0.62, 0.86))  # ablation -> dataset
    arrow((0.5, 0.79), (0.62, 0.68))  # ablation -> wrapper
    arrow((0.84, 0.74), (0.84, 0.74))  # small connector (not used)
    arrow((0.72, 0.74), (0.82, 0.68))
    arrow((0.78, 0.68), (0.84, 0.68))
    arrow((0.82, 0.62), (0.82, 0.5))
    arrow((0.7, 0.5), (0.62, 0.5))

    # From pipeline to metrics & results
    arrow((0.84, 0.56), (0.86, 0.5))
    arrow((0.86, 0.46), (0.86, 0.34))
    arrow((0.7, 0.42), (0.67, 0.34))
    arrow((0.5, 0.3), (0.5, 0.2))
    arrow((0.5, 0.16), (0.5, 0.08))

    ax.set_title('High-level Experiment Flow', fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_ablation_examples(path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    start_x = 0.05
    start_y = 0.72
    w = 0.24
    h = 0.12

    def step_box(x, y, text, col='#ffffff'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2', edgecolor='black', facecolor=col)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

    # Full pipeline
    step_box(start_x, start_y, 'Imputer')
    step_box(start_x + 0.26, start_y, 'Scaler')
    step_box(start_x + 0.52, start_y, 'PCA')
    step_box(start_x + 0.78, start_y, 'GridSearchCV(LGBM)')
    step_box(start_x + 1.04, start_y, 'VotingRegressor')
    ax.text(start_x, start_y + 0.14, 'Full', fontsize=11, weight='bold')

    # Arrows
    for i in range(4):
        ax.annotate('', xy=(start_x + 0.26*(i + 0.5) + w*0.5/i if i!=0 else start_x + 0.79, start_y + h/2), xytext=(start_x + 0.26*i + w*0.5, start_y + h/2), arrowprops=dict(arrowstyle='->'))

    # No scaling
    y2 = start_y - 0.25
    step_box(start_x, y2, 'Imputer')
    step_box(start_x + 0.26, y2, 'Identity')
    step_box(start_x + 0.52, y2, 'PCA')
    step_box(start_x + 0.78, y2, 'GridSearchCV(LGBM)')
    step_box(start_x + 1.04, y2, 'VotingRegressor')
    ax.text(start_x, y2 + 0.14, 'No scaling (keeps imputer)', fontsize=11, weight='bold')

    # No tuning
    y3 = y2 - 0.25
    step_box(start_x, y3, 'Imputer')
    step_box(start_x + 0.26, y3, 'Scaler')
    step_box(start_x + 0.52, y3, 'PCA')
    step_box(start_x + 0.78, y3, 'LGBMRegressor (estimator)')
    step_box(start_x + 1.04, y3, 'VotingRegressor')
    ax.text(start_x, y3 + 0.14, 'No tuning (GridSearchCV -> estimator)', fontsize=11, weight='bold')

    # No ensemble
    y4 = y3 - 0.25
    step_box(start_x, y4, 'Imputer')
    step_box(start_x + 0.26, y4, 'Scaler')
    step_box(start_x + 0.52, y4, 'PCA')
    step_box(start_x + 0.78, y4, 'GridSearchCV(LGBM)')
    step_box(start_x + 1.04, y4, 'LinearRegression (fallback)')
    ax.text(start_x, y4 + 0.14, 'No ensemble (ensemble -> fallback estimator)', fontsize=11, weight='bold')

    ax.set_title('Ablation Examples: Step Changes', fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    out1 = OUT_DIR / 'high_level_flow.png'
    out2 = OUT_DIR / 'ablation_examples.png'
    draw_flowchart(out1)
    draw_ablation_examples(out2)
    print('Wrote diagrams:', out1, out2)
