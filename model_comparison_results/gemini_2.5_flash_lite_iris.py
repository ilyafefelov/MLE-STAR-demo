"""
Model: gemini-2.5-flash-lite
Dataset: iris
Generated: 2025-11-13T12:57:49.618040
"""

def build_full_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    # Preprocessing steps: Imputation and Scaling
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Feature Engineering: Dimensionality Reduction
    # For a dataset with only 4 features and 150 samples, PCA can be useful
    # to reduce dimensionality and potentially noise, even if the number of features is small.
    # We'll aim to reduce to 2 components as it's a common practice for visualization
    # and can sometimes improve model performance by focusing on dominant variations.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=2, random_state=42))
    ])

    # Model Selection and Hyperparameter Tuning
    # Given the small dataset size (150 samples) and number of features (4),
    # models that are less prone to overfitting and can capture non-linear relationships
    # are good candidates.
    # - SVC with RBF kernel can capture non-linearities.
    # - RandomForestClassifier is robust and good for small datasets.
    # - MLPClassifier can learn complex patterns but might overfit if not regularized.
    # - GradientBoostingClassifier is powerful but can overfit on small data.
    #
    # We will consider SVC as a sophisticated choice for this task.
    # The RBF kernel is flexible for non-linear decision boundaries.
    # For hyperparameter tuning, we'll focus on C (regularization parameter) and gamma (kernel coefficient).
    # Given the small dataset, we need to be careful with the search space.

    model_svc = SVC(random_state=42, probability=True) # probability=True is useful for some metrics
    param_grid_svc = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale'], # 'scale' is often a good default
        'kernel': ['rbf']
    }
    grid_search_svc = GridSearchCV(model_svc, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)

    # We'll choose SVC as the best model for this example due to its ability to handle
    # non-linear decision boundaries and its general robustness.
    # The GridSearch will find the best hyperparameters for SVC on this dataset.

    # Uncomment and consider other models if SVC doesn't perform as expected or for comparison:
    #
    # model_rf = RandomForestClassifier(random_state=42)
    # param_grid_rf = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5, 10]
    # }
    # grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    #
    # model_gb = GradientBoostingClassifier(random_state=42)
    # param_grid_gb = {
    #     'n_estimators': [50, 100, 150],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7]
    # }
    # grid_search_gb = GridSearchCV(model_gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
    #
    # model_mlp = MLPClassifier(random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1)
    # param_grid_mlp = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': [0.0001, 0.001, 0.01]
    # }
    # grid_search_mlp = GridSearchCV(model_mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)

    # For this implementation, we will use the tuned SVC from GridSearchCV
    model = grid_search_svc

    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])