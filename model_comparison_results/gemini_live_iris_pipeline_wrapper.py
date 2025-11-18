from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier



def build_full_pipeline(random_state=42):
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    model = HistGradientBoostingClassifier(random_state=random_state)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline