from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # For dimensionality reduction on a dataset with 178 samples and 13 features,
    # PCA can be a good choice to reduce overfitting and potentially improve
    # model performance by capturing the most significant variance.
    # We'll aim to retain a good portion of the variance.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=random_state))
    ])

    # Model Selection Rationale:
    # Dataset size: 178 samples is relatively small. Models that are prone to
    # overfitting (like very complex deep neural networks without regularization
    # or large random forests) might struggle.
    # Features: 13 features is a moderate number.
    # Classes: 3 classes are well-separated in the wine dataset.
    #
    # Considering these factors and the requirement for a sophisticated model:
    # - LogisticRegression: While simple, it can be a good baseline but might not
    #   capture complex non-linear relationships.
    # - SVC: With a suitable kernel (like 'rbf'), it can handle non-linearities
    #   and often performs well on small to medium-sized datasets. Hyperparameter
    #   tuning is crucial.
    # - RandomForestClassifier: Can be powerful but might overfit on small datasets
    #   if not well-tuned.
    # - GradientBoostingClassifier: Often robust and performs well, but can also
    #   overfit if not managed.
    # - MLPClassifier: Can learn complex patterns but is very prone to overfitting
    #   on small datasets and requires careful tuning of architecture and regularization.
    #
    # Based on typical performance on the wine dataset and considering the need for
    # a sophisticated yet manageable model for this dataset size, SVC with an RBF kernel
    # and well-tuned hyperparameters is a strong candidate. GradientBoostingClassifier
    # is also a good choice. We will tune SVC for this example.

    # Hyperparameter tuning for SVC
    param_grid_svc = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf'],
        'random_state': [42]
    }
    svc_model = SVC()
    grid_search_svc = GridSearchCV(svc_model, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)

    # Using SVC as the chosen model for its ability to handle non-linear decision
    # boundaries effectively with a radial basis function kernel, and it generally
    # performs well on datasets of this size without excessive overfitting when tuned.
    model = grid_search_svc

    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])