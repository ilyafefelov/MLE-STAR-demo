import pytest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.mle_star_ablation.mle_star_generated_pipeline import (
    set_full_pipeline_callable,
    build_pipeline,
    inspect_pipeline,
    is_ablation_meaningful,
)
from src.mle_star_ablation.config import AblationConfig


def sample_builder_with_synonyms(random_state=42):
    pre = Pipeline([("imputer", SimpleImputer(strategy='mean')), ("scaler", StandardScaler())])
    fe = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])
    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    return Pipeline([("scaler", pre), ("feature_pipeline", fe), ("estimator", model)])


def test_normalize_and_ablation_toggles():
    # Register custom builder
    set_full_pipeline_callable(sample_builder_with_synonyms)
    cfg_full = AblationConfig(name='full')
    pipe_full = build_pipeline(cfg_full, deterministic=True, random_state=42)
    info = inspect_pipeline(pipe_full)
    # After normalization, pipeline should be canonicalized and contain steps
    assert 'model' in info['steps'] or any(s.startswith('model') for s in info['steps'])
    assert info['has_scaler']
    assert info['has_feature_engineering']
    # Ensure ablation meaningfulness detects components
    assert is_ablation_meaningful(pipe_full)

    # Now test turning off feature engineering
    cfg_no_fe = AblationConfig(name='no_feature_engineering', use_feature_engineering=False)
    pipe_no_fe = build_pipeline(cfg_no_fe, deterministic=True, random_state=42)
    info_no_fe = inspect_pipeline(pipe_no_fe)
    assert not info_no_fe['has_feature_engineering']

    # Turn off scaling
    cfg_no_scaling = AblationConfig(name='no_scaling', use_scaling=False)
    pipe_no_scaling = build_pipeline(cfg_no_scaling, deterministic=True, random_state=42)
    info_no_scaling = inspect_pipeline(pipe_no_scaling)
    assert not info_no_scaling['has_scaler']
