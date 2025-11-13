"""
Model: gemini-2.5-pro
Dataset: iris
Generated: 2025-11-13T12:58:44.516826
"""

def build_full_pipeline():
    """
    Builds a complete scikit-learn machine learning pipeline for the Iris dataset.

    The pipeline includes preprocessing, feature engineering (PCA), and a tuned
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
    # This sub-pipeline handles missing values (though Iris has none, it's good practice)
    # and scales the data. Scaling is crucial for distance-based algorithms like PCA and SVC.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Step 2: Feature Engineering
    # We use Principal Component Analysis (PCA) for dimensionality reduction.
    # For a dataset with 4 correlated features, reducing to 2 components can
    # capture most of the variance, reduce noise, and make the classification
    # task easier for the model.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=2, random_state=42))
    ])

    # Step 3: Model Selection and Hyperparameter Tuning
    #
    # Model Choice: Support Vector Classifier (SVC)
    # - Dataset Size (150 samples): SVCs are highly effective on smaller datasets
    #   where they can find a clear separating margin without needing large amounts of data.
    # - Features (4, reduced to 2): SVCs excel in low-dimensional spaces.
    #   The kernel trick allows them to find non-linear boundaries efficiently.
    # - Classes (3): SVCs handle multi-class problems natively using a one-vs-one
    #   or one-vs-rest strategy.
    #
    # An SVC is a sophisticated and powerful choice that is well-suited to the
    # geometric nature of the Iris classification problem.
    #
    # Hyperparameters:
    # - kernel='rbf': The Radial Basis Function kernel is a good default for
    #   capturing complex, non-linear relationships.
    # - C=10: A moderately high regularization parameter. It creates a "harder"
    #   margin, fitting the training data more closely, which works well for
    #   this clean, well-separated dataset.
    # - gamma='auto': A good starting point for the kernel coefficient that adapts
    #   to the number of features.
    # - probability=True: Enables probability estimates for predictions.
    model = SVC(kernel='rbf', C=10, gamma='auto', probability=True, random_state=42)

    # Combine all steps into the final pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])