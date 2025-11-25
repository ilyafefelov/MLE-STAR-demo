
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

print(f"Sklearn version: {sklearn.__version__}")

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
scaler = StandardScaler()
out = scaler.fit_transform(df)
print(f"Output type: {type(out)}")
if hasattr(out, 'columns'):
    print(f"Columns: {out.columns}")

try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
except ImportError:
    print("LightGBM not installed")
