"""
Model: gemini-2.5-pro
Dataset: wine
Generated: 2025-11-13T12:55:32.995885
"""

def build_full_pipeline():
    """
    Builds a complete scikit-learn ML pipeline for the 'wine' dataset.

    The pipeline includes preprocessing, feature engineering (dimensionality reduction),
    and a fine-tuned classification model.

    Returns:
        sklearn.pipeline.Pipeline: A complete, configured scikit-learn pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC

    # Step 1: Preprocessor
    # A nested pipeline to handle missing values and feature scaling.
    # - SimpleImputer: Replaces any missing values with the mean of the column.
    # - StandardScaler: Scales features to have zero mean and unit variance,
    #   which is crucial for distance-based algorithms like SVC.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Step 2: Feature Engineering
    # Dimensionality reduction using Linear Discriminant Analysis (LDA).
    # LDA is a supervised technique that projects the data to a lower-dimensional
    # space while maximizing the separation between classes. For a 3-class
    # problem, it reduces the features to at most (n_classes - 1) = 2 components,
    # which is ideal for simplifying the model without losing class-discriminatory information.
    feature_engineering = Pipeline([
        ('lda', LinearDiscriminantAnalysis(n_components=2))
    ])

    # Step 3: Model Selection and Hyperparameter Tuning
    #
    # Model Choice: Support Vector Classifier (SVC)
    # - Dataset Size (178 samples): SVCs are highly effective in high-dimensional
    #   spaces and are memory efficient, making them one of the best choices for
    #   small-to-medium datasets where they are less prone to overfitting
    #   compared to complex ensembles like GradientBoostingClassifier or deep
    #   learning models like MLPClassifier.
    # - Features (13) & Classes (3): The kernel trick allows SVC to model
    #   complex, non-linear relationships between the 13 features to separate the 3 classes.
    # - Sophistication: It is a powerful and sophisticated model, moving beyond
    #   simple linear boundaries.
    #
    # Hyperparameters:
    # - C=10: A moderately high regularization parameter. It allows for a smaller-margin
    #   hyperplane, fitting the training data more accurately, which is reasonable
    #   for a small, relatively clean dataset.
    # - kernel='rbf': The Radial Basis Function kernel is a good default for
    #   capturing complex, non-linear patterns.
    # - gamma=0.1: Defines how much influence a single training example has. A value
    #   of 0.1 is a balanced choice, preventing the model from being either too
    #   constrained or too flexible.
    # - probability=True: Enables probability estimates for predictions.
    model = SVC(C=10, kernel='rbf', gamma=0.1, probability=True, random_state=42)

    # Final Full Pipeline
    # Chains the preprocessing, feature engineering, and modeling steps together.
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])