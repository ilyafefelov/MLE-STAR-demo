from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def build_full_pipeline():
    preprocessor = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    return pipeline

if __name__ == '__main__':
    # Example train code (should not run in the wrapper)
    pass
