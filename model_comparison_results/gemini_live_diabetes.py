import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder

def build_full_pipeline():
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer())
    ])

    # Preprocessing for categorical data (if any, though diabetes is mostly numeric)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    # Diabetes dataset is all numeric, but we keep the structure generic
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, slice(0, 10)), # First 10 columns are features
        ],
        remainder='drop'
    )

    # Feature Selection
    feature_selection = SelectFromModel(GradientBoostingRegressor(n_estimators=50, random_state=42))

    # Ensemble Model
    reg1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg2 = RandomForestRegressor(n_estimators=100, random_state=42)
    reg3 = Ridge(alpha=1.0)

    ensemble = VotingRegressor(estimators=[
        ('gb', reg1), 
        ('rf', reg2), 
        ('ridge', reg3)
    ])

    # Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_selection),
        ('model', ensemble)
    ])

    return pipeline

if __name__ == "__main__":
    pipeline = build_full_pipeline()
    print("Pipeline built successfully")
