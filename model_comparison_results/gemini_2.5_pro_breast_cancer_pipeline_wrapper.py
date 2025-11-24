from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
def build_full_pipeline(random_state: int = 42):
    """
    Builds a complete scikit-learn ML pipeline for the breast_cancer dataset.

    The pipeline includes preprocessing, feature engineering (dimensionality reduction),
    and a sophisticated, tuned classification model.

    Returns:
        sklearn.pipeline.Pipeline: The fully constructed ML pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    # Step 1: Preprocessor
    # A sub-pipeline for data cleaning and scaling.
    # - SimpleImputer handles any potential missing values (good practice).
    # - StandardScaler standardizes features to have zero mean and unit variance,
    #   which is crucial for distance-based algorithms like PCA and SVC.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Step 2: Feature Engineering
    # A sub-pipeline for dimensionality reduction.
    # - PCA (Principal Component Analysis) is used to reduce the 30 features
    #   to a smaller set of 10 principal components. This helps to reduce noise,
    #   prevent overfitting, and can improve model performance and training speed,
    #   which is beneficial for a dataset with a relatively small number of samples (569).
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=10, random_state=random_state))
    ])

    # Step 3: Model Selection and Hyperparameter Tuning
    # - Dataset Analysis:
    #   - Samples: 569 (small)
    #   - Features: 30 (moderately high-dimensional relative to sample size)
    #   - Classes: 2 (binary classification)
    # - Model Choice: SVC (Support Vector Classifier)
    #   - SVC with an RBF kernel is a powerful, sophisticated model capable of finding
    #     complex, non-linear decision boundaries.
    #   - It is particularly effective in high-dimensional spaces and is less prone
    #     to overfitting on smaller datasets compared to complex ensembles or neural networks.
    #   - Its performance is highly dependent on proper feature scaling, which is
    #     handled by the 'preprocessor' step.
    # - Hyperparameter Tuning:
    #   - C=10: A moderately high regularization parameter. It allows the model to
    #     fit the training data more closely, aiming for a lower bias. This value
    #     often works well on this specific dataset.
    #   - kernel='rbf': The Radial Basis Function kernel is chosen to handle
    #     non-linear relationships between features.
    #   - gamma='auto': The kernel coefficient is set to 'auto' (1 / n_features),
    #     a reasonable default for capturing feature influence.
    #   - probability=True: Enables the model to predict class probabilities,
    #     which is useful for more detailed evaluation (e.g., ROC AUC).
    model = SVC(C=10, kernel='rbf', gamma='auto', probability=True, random_state=random_state)

    # Combine all steps into a single, final pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])