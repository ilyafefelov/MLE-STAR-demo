from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    preprocessor = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    return pipeline