from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_full_pipeline(random_state=42):
    # Diabetes dataset is small (442 samples), so we need to be careful with overfitting.
    # SimpleImputer is good practice even if data is clean.
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    
    feature_engineering = None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        PCA = None
    
    # For small datasets, PCA might lose too much info if not careful, but let's keep it as an option
    # or maybe disable it. The agent usually includes it.
    if PCA is not None:
        feature_engineering = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])
    
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError('lightgbm not installed. Install it with: pip install lightgbm')
    
    # Base model
    model = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
    
    param_grid = None
    try:
        from sklearn.model_selection import GridSearchCV
    except Exception:
        GridSearchCV = None
    
    if GridSearchCV is not None:
        # Smaller num_leaves for smaller dataset to prevent overfitting
        param_grid = {
            'num_leaves': [7, 15, 31], 
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100]
        }
        model = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=1)
    
    try:
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import Ridge
    except Exception:
        VotingRegressor = None
    
    if VotingRegressor is not None:
        # Ensemble with a linear model is often good for diabetes dataset
        ensemble = VotingRegressor([("lgbm", model), ("ridge", Ridge())])
        model = ensemble
    
    steps = [("preprocessor", preprocessor)]
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    steps.append(("model", model))
    
    pipeline = Pipeline(steps)
    return pipeline
