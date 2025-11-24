#!/usr/bin/env python
"""
Generate high-quality diagrams using Matplotlib with precise layout.
Creates `docs/diagrams/high_level_flow_refined.png/svg` and `docs/diagrams/ablation_examples_refined.png/svg`.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'diagrams'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOX_STYLE = dict(boxstyle='round,pad=0.4', linewidth=1.2, edgecolor='#333333')


def draw_box(ax, x, y, w, h, text, fc='#ffffff', fontsize=10):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.4', fc=fc, ec='#333333')
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)
    return rect


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', linewidth=1.2, color='#333333'))


def generate_high_level(path_png, path_svg):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    xs = [0.1, 0.32, 0.68, 0.9]
    y = 0.7
    w = 0.2
    h = 0.15

    draw_box(ax, xs[0], y, w, h, 'run_experiment_suite.py\n(manifest)')
    draw_box(ax, xs[1], y, w, h, 'run_ablation.py\n(ablation loops)')
    draw_box(ax, xs[2], y, w, h, 'Pipeline Wrapper\n(model_comparison_results)')
    draw_box(ax, xs[3], y, w, h, 'Adapter\n(mle_star_generated_pipeline.py)')

    # dataset and pipeline lower row
    ys2 = 0.3
    draw_box(ax, 0.32, ys2, w, h, 'DatasetLoader\n(load & split)')
    draw_box(ax, 0.68, ys2, w, h, 'Sklearn Pipeline\n(fit/predict)')
    # metrics box
    draw_box(ax, 0.5, 0.05, 0.6, 0.12, 'Metrics + Analysis\n(summary, pairwise, plots)')

    # arrows
    arrow(ax, 0.2, 0.7, 0.28, 0.7)
    arrow(ax, 0.38, 0.7, 0.58, 0.7)
    arrow(ax, 0.78, 0.7, 0.86, 0.7)

    arrow(ax, 0.38, 0.65, 0.38, 0.42)
    arrow(ax, 0.62, 0.65, 0.62, 0.42)
    arrow(ax, 0.68, 0.42, 0.62, 0.42)
    arrow(ax, 0.5, 0.17, 0.5, 0.12)

    ax.set_title('High-level Experiment Flow', fontsize=14)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_svg)
    plt.close(fig)


def generate_ablation_examples(path_png, path_svg):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # columns: steps
    step_centers = [0.15, 0.33, 0.51, 0.69, 0.87]
    step_names = ['Imputer', 'Scaler', 'PCA', 'GridSearchCV(Est)', 'VotingRegressor']

    # title row
    y_full = 0.85
    for i, (cx, name) in enumerate(zip(step_centers, step_names)):
        draw_box(ax, cx, y_full, 0.18, 0.12, name, fc='#f7fbff')
    ax.text(0.03, y_full + 0.08, 'Full', fontsize=11, fontweight='bold')

    # no_scaling row
    y_ns = 0.6
    draw_box(ax, step_centers[0], y_ns, 0.18, 0.12, 'Imputer')
    draw_box(ax, step_centers[1], y_ns, 0.18, 0.12, 'Identity\n(no scaler)', fc='#fff7f0')
    draw_box(ax, step_centers[2], y_ns, 0.18, 0.12, 'PCA')
    draw_box(ax, step_centers[3], y_ns, 0.18, 0.12, 'GridSearchCV(Est)')
    draw_box(ax, step_centers[4], y_ns, 0.18, 0.12, 'VotingRegressor')
    ax.text(0.03, y_ns + 0.05, 'No scaling', fontsize=11, fontweight='bold')

    # no_tuning row
    y_nt = 0.35
    draw_box(ax, step_centers[0], y_nt, 0.18, 0.12, 'Imputer')
    draw_box(ax, step_centers[1], y_nt, 0.18, 0.12, 'Scaler')
    draw_box(ax, step_centers[2], y_nt, 0.18, 0.12, 'PCA')
    draw_box(ax, step_centers[3], y_nt, 0.18, 0.12, 'Est (no Grid)')
    draw_box(ax, step_centers[4], y_nt, 0.18, 0.12, 'VotingRegressor')
    ax.text(0.03, y_nt + 0.05, 'No tuning', fontsize=11, fontweight='bold')

    # no_ensemble row
    y_ne = 0.1
    draw_box(ax, step_centers[0], y_ne, 0.18, 0.12, 'Imputer')
    draw_box(ax, step_centers[1], y_ne, 0.18, 0.12, 'Scaler')
    draw_box(ax, step_centers[2], y_ne, 0.18, 0.12, 'PCA')
    draw_box(ax, step_centers[3], y_ne, 0.18, 0.12, 'GridSearchCV(Est)')
    draw_box(ax, step_centers[4], y_ne, 0.18, 0.12, 'LinearReg (fallback)', fc='#fff7f0')
    ax.text(0.03, y_ne + 0.05, 'No ensemble', fontsize=11, fontweight='bold')

    ax.set_title('Ablation Examples: Step Changes', fontsize=14)

    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_svg)
    plt.close(fig)


if __name__ == '__main__':
    out1_png = OUT_DIR / 'high_level_flow_refined.png'
    out1_svg = OUT_DIR / 'high_level_flow_refined.svg'
    out2_png = OUT_DIR / 'ablation_examples_refined.png'
    out2_svg = OUT_DIR / 'ablation_examples_refined.svg'
    generate_high_level(out1_png, out1_svg)
    generate_ablation_examples(out2_png, out2_svg)
    print('Wrote diagrams:', out1_png, out1_svg, out2_png, out2_svg)
