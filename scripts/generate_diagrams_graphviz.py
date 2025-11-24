#!/usr/bin/env python
"""
Generate high-quality diagrams using Graphviz (python-graphviz).
- Creates `docs/diagrams/high_level_flow_graphviz.svg` and PNG
- Creates `docs/diagrams/ablation_examples_graphviz.svg` and PNG

If Graphviz or python-graphviz is not installed, fallback to text DOT files and exit.
"""
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'diagrams'
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from graphviz import Digraph
except Exception as e:
    print('Graphviz python package not found. Install with `pip install graphviz` (and graphviz system package).')
    raise


def generate_high_level(out_svg: Path, out_png: Path):
    g = Digraph('ExperimentFlow', format='svg')
    g.attr(rankdir='LR', fontsize='12')
    g.attr('node', shape='box', style='rounded,filled', fillcolor='#fdfdfd', color='#2f2f2f', fontname='Helvetica')

    g.node('suite', 'run_experiment_suite.py\n(manifest)')
    g.node('ablation', 'run_ablation.py\n(ablation loops)')
    g.node('dataset', 'DatasetLoader\n(src/mle_star_ablation/datasets.py)')
    g.node('wrapper', 'Pipeline Wrapper\n(model_comparison_results/*_pipeline_wrapper.py)')
    g.node('adapter', 'Adapter\n(src/mle_star_ablation/mle_star_generated_pipeline.py)')
    g.node('pipeline', 'Sklearn Pipeline\n(fit / predict)')
    g.node('metrics', 'Metrics & Analysis\n(summary, pairwise)')
    g.node('results', 'results/<exp>/*\n(csv, png, txt)')

    g.edge('suite', 'ablation')
    g.edge('ablation', 'dataset')
    g.edge('ablation', 'wrapper', label='imports/ registers')
    g.edge('wrapper', 'adapter', label='build_full_pipeline()')
    g.edge('adapter', 'pipeline', label='build_pipeline(config)')
    g.edge('pipeline', 'metrics')
    g.edge('metrics', 'results')

    # Save
    g.render(filename=str(out_svg.with_suffix('')), cleanup=True)
    # convert to png via render
    g.format = 'png'
    g.render(filename=str(out_png.with_suffix('')), cleanup=True)


def generate_ablation_examples(out_svg: Path, out_png: Path):
    g = Digraph('AblationExamples', format='svg')
    g.attr(rankdir='LR', fontsize='12')
    g.attr('node', shape='rect', style='rounded,filled', fillcolor='#ffffff', fontname='Helvetica')

    # Full pipeline
    g.node('imputer', 'Imputer')
    g.node('scaler', 'Scaler')
    g.node('pca', 'PCA')
    g.node('grid', 'GridSearchCV(Estimator)')
    g.node('ensemble', 'VotingRegressor')
    g.edge('imputer', 'scaler')
    g.edge('scaler', 'pca')
    g.edge('pca', 'grid')
    g.edge('grid', 'ensemble')

    # No scaling variant cluster
    g.attr('node', shape='rect', style='dashed,rounded')
    g.node('imputer_ns', 'Imputer')
    g.node('identity', 'Identity (no scaler)')
    g.node('pca_ns', 'PCA')
    g.node('grid_ns', 'GridSearchCV(Estimator)')
    g.node('ensemble_ns', 'VotingRegressor')
    g.edge('imputer_ns', 'identity')
    g.edge('identity', 'pca_ns')
    g.edge('pca_ns', 'grid_ns')
    g.edge('grid_ns', 'ensemble_ns')

    # No tuning variant cluster
    g.attr('node', shape='rect', style='dotted,rounded')
    g.node('imputer_nt', 'Imputer')
    g.node('scaler_nt', 'Scaler')
    g.node('pca_nt', 'PCA')
    g.node('est_nt', 'Estimator (no Grid)')
    g.node('ensemble_nt', 'VotingRegressor')
    g.edge('imputer_nt', 'scaler_nt')
    g.edge('scaler_nt', 'pca_nt')
    g.edge('pca_nt', 'est_nt')
    g.edge('est_nt', 'ensemble_nt')

    # No ensemble variant cluster
    g.attr('node', shape='rect', style='rounded,filled', fillcolor='#ffffff')
    g.node('imputer_ne', 'Imputer')
    g.node('scaler_ne', 'Scaler')
    g.node('pca_ne', 'PCA')
    g.node('grid_ne', 'GridSearchCV(Estimator)')
    g.node('fallback', 'Estimator (fallback)')
    g.edge('imputer_ne', 'scaler_ne')
    g.edge('scaler_ne', 'pca_ne')
    g.edge('pca_ne', 'grid_ne')
    g.edge('grid_ne', 'fallback')

    g.render(filename=str(out_svg.with_suffix('')), cleanup=True)
    g.format = 'png'
    g.render(filename=str(out_png.with_suffix('')), cleanup=True)


if __name__ == '__main__':
    out1_svg = OUT_DIR / 'high_level_flow_graphviz.svg'
    out1_png = OUT_DIR / 'high_level_flow_graphviz.png'
    out2_svg = OUT_DIR / 'ablation_examples_graphviz.svg'
    out2_png = OUT_DIR / 'ablation_examples_graphviz.png'
    generate_high_level(out1_svg, out1_png)
    generate_ablation_examples(out2_svg, out2_png)
    print('Wrote diagrams:', out1_svg, out1_png, out2_svg, out2_png)
