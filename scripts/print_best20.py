import pandas as pd
from pathlib import Path
p = Path('results/aggregated_summary.csv')
df = pd.read_csv(p)
# keep only n_runs == 20
best = df[df['n_runs']==20].groupby('experiment').apply(lambda g: g.loc[g['mean'].idxmax()])
best = best.reset_index(drop=True)
cols = ['experiment','configuration','mean','std','ci_lower','ci_upper','n_runs']
print(best[cols].to_csv(index=False))