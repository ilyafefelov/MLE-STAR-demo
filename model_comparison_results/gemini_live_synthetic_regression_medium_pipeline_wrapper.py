from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_full_pipeline(random_state=42):
    # Synthetic Medium: More noise or complexity.
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    try:
        preprocessor.set_output(transform="pandas")
    except AttributeError:
        pass
    
    feature_engineering = None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        PCA = None
    if PCA is not None:
        feature_engineering = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])
    
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError('lightgbm not installed. Install it with: pip install lightgbm')
    
    model = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
    
    param_grid = None
    try:
        from sklearn.model_selection import GridSearchCV
    except Exception:
        GridSearchCV = None
    
    if GridSearchCV is not None:
        # Allow for more complexity
        param_grid = {
            'num_leaves': [31, 63, 127], 
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }
        model = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=1)
    
    steps = [("preprocessor", preprocessor)]
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    return pipeline
