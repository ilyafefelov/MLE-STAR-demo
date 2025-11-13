"""
Model: gemini-2.5-flash
Dataset: iris
Generated: 2025-11-13T12:58:15.054606
"""

def build_full_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    # No need for LogisticRegression, RandomForestClassifier, SVC, MLPClassifier
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # For the Iris dataset (4 features, 150 samples), PCA is a suitable
    # dimensionality reduction technique. Reducing to 2 components allows for
    # capturing the most variance while significantly reducing complexity,
    # which can sometimes help models generalize better or simplify the feature space.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=2, random_state=42))
    ])
    
    # Model Choice Rationale:
    # - Dataset size (150 samples): Small datasets can be prone to overfitting with overly complex models.
    #   However, ensemble methods like Gradient Boosting, when appropriately tuned, can perform very well.
    # - Features (4): Very few features. Gradient Boosting can effectively learn complex
    #   relationships even with a limited number of features.
    # - Classes (3): Multiclass classification is handled natively by GradientBoostingClassifier.
    # - Sophistication: GradientBoostingClassifier is a powerful and sophisticated ensemble
    #   technique known for its high predictive accuracy. It builds trees sequentially,
    #   each correcting errors of the previous ones.
    # - Hyperparameter Tuning: For a small dataset like Iris, slightly lower learning rates,
    #   fewer estimators, and shallower trees (smaller max_depth) compared to default
    #   settings often help prevent overfitting while maintaining strong performance.
    #   We choose specific values for n_estimators, learning_rate, and max_depth
    #   to fine-tune for this dataset's characteristics.
    model = GradientBoostingClassifier(
        n_estimators=70,         # A slightly reduced number of estimators for a smaller dataset
        learning_rate=0.07,      # A slightly reduced learning rate for better generalization
        max_depth=3,             # A shallow tree depth to prevent overfitting
        random_state=42
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])