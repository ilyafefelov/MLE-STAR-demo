from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
def build_full_pipeline(random_state: int = 42):
    """
    Builds a complete scikit-learn ML pipeline for the digits dataset.

    This pipeline includes preprocessing, feature engineering (PCA), and a tuned
    Support Vector Classifier (SVC) model.

    Returns:
        sklearn.pipeline.Pipeline: The complete, configured ML pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    # Step 1: Preprocessor
    # A sub-pipeline for data cleaning and scaling.
    # - SimpleImputer handles potential missing values (good practice, though digits has none).
    # - StandardScaler standardizes features by removing the mean and scaling to unit variance,
    #   which is crucial for distance-based algorithms like PCA and SVC.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Step 2: Feature Engineering
    # A sub-pipeline for dimensionality reduction.
    # - PCA (Principal Component Analysis) is used to reduce the 64-dimensional pixel
    #   space into a lower-dimensional space, capturing the most significant variance.
    #   This helps to reduce noise, speed up training, and can improve model generalization.
    #   The number of components is chosen to retain significant information while reducing complexity.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=40, random_state=random_state))
    ])

    # Step 3: Model Selection and Tuning
    # The Support Vector Classifier (SVC) is chosen as the final model.
    # - Dataset Size (1797 samples): SVC is highly effective on smaller datasets
    #   where it can find a clear separating hyperplane without overfitting.
    # - Features (64, reduced to 40): SVC excels in high-dimensional spaces.
    #   The RBF kernel allows it to learn complex, non-linear decision boundaries,
    #   which is perfect for image-like pixel data.
    # - Classes (10): SVC natively handles binary classification, but scikit-learn
    #   implements a "one-vs-one" strategy for multi-class problems, which is
    #   efficient and performs well here.
    #
    # Hyperparameters have been tuned for this specific dataset:
    # - C=10: A moderately high regularization parameter, allowing the model to fit
    #   the training data well, trusting that the PCA has reduced noise.
    # - gamma='scale': An appropriate kernel coefficient that scales with the
    #   number of features and data variance.
    # - kernel='rbf': The Radial Basis Function kernel is ideal for capturing the
    #   complex patterns in the digit shapes.
    # - probability=True: Enables probability estimates for predictions.
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=random_state)

    # Combine all steps into the final full pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])