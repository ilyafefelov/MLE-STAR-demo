from pathlib import Path
import pandas as pd
p = Path('results/aggregated_summary.csv')
df = pd.read_csv(p)
# Filter for n_runs == 20
df20 = df[df['n_runs']==20]
experiments = df20['experiment'].unique()
out_lines = []
for exp in experiments:
    out_lines.append(f"### {exp}")
    sub = df20[df20['experiment']==exp]
    # Select columns
    cols = ['configuration','mean','std','ci_lower','ci_upper','n_runs']
    out_lines.append('| Configuration | Mean | Std | CI lower | CI upper | n_runs |')
    out_lines.append('|---|---:|---:|---:|---:|---:|')
    for _, r in sub[cols].iterrows():
        out_lines.append(f"| {r['configuration']} | {r['mean']:.6f} | {r['std']:.6f} | {r['ci_lower']:.6f} | {r['ci_upper']:.6f} | {int(r['n_runs'])} |")
    # Add links to common PNGs if present
    base = Path('results')/exp.split('/')[0]/exp.split('/')[1]
    # common filenames
    imgs = ['statistical_overview.png','comparison_boxplot.png','pvalue_heatmap.png','tstat_heatmap.png','zscore_heatmap.png']
    for img in imgs:
        imgp = base/img
        if imgp.exists():
            out_lines.append(f"![{img}]({imgp.as_posix()})")
    out_lines.append('\n')

out = '\n'.join(out_lines)
Path('reports/detailed_stats.md').write_text(out)
print('Wrote reports/detailed_stats.md')
