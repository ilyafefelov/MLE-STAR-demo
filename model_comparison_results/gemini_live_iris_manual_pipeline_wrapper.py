from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

def build_full_pipeline(random_state=42):
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    feature_engineering = None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        PCA = None
    if PCA is not None:
        feature_engineering = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])
    model = HistGradientBoostingClassifier(random_state=random_state)
    param_grid = None
    try:
        from sklearn.model_selection import GridSearchCV
    except Exception:
        GridSearchCV = None
    if GridSearchCV is not None:
        param_grid = {'max_depth': [3, None], 'learning_rate': [0.05, 0.1]}
        model = GridSearchCV(model, param_grid=param_grid, cv=2, n_jobs=1)
    try:
        from sklearn.ensemble import VotingRegressor, VotingClassifier
    except Exception:
        VotingRegressor = None; VotingClassifier = None
    if VotingRegressor is not None or VotingClassifier is not None:
        from sklearn.linear_model import LogisticRegression
        ensemble = VotingClassifier([("m1", model), ("m2", LogisticRegression())], voting="soft")
        model = ensemble
    steps = [("preprocessor", preprocessor)]
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    return pipeline