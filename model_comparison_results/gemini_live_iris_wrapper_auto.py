from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.datasets import load_iris


def build_full_pipeline(random_state=42):
    hgb_model = HistGradientBoostingClassifier(random_state=42)

    # Build and return a scikit-learn Pipeline using detected components
    try:
        if "pipeline" in locals():
            return pipeline
    except Exception:
        pass

    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    model = hgb_model if "hgb_model" in globals() or "hgb_model" in locals() else None
    if model is None:
        raise RuntimeError("Could not detect model or pipeline from generated script")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline