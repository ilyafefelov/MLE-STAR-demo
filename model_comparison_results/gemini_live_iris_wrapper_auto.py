import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.datasets import load_iris


def build_full_pipeline(random_state=42):
    hgb_model = HistGradientBoostingClassifier(random_state=42)
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7, None],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=hgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # Return the inner estimator template, not a fitted object
    try:
        if "grid_search" in locals():
            return grid_search.estimator
        if "hgb_model" in locals():
            return hgb_model
        if "clf" in locals():
            return clf
        # No fitted 'best_model' available; fallback to available unfitted estimator
        raise RuntimeError('No fitted best_model available; returning registry-estimator instead')
    except Exception:
        # Fallback: return the pipeline variable if present
        try:
            return pipeline
        except Exception:
            raise RuntimeError("No pipeline-like variable found")