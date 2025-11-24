from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    # Requirement 1: Preprocessor pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Requirement 1: Feature Engineering pipeline
    # For a dataset with 13 features and 178 samples, PCA can help in dimensionality reduction,
    # reducing noise and potentially improving model generalization, especially for models
    # like SVC which can sometimes benefit from a more compact feature space.
    # We choose to reduce to 8 components, aiming to capture most variance while reducing dimensionality.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=8, random_state=random_state))
    ])

    # Requirement 1 & 5 & 6: Choose BEST sophisticated model with tuned hyperparameters
    # Requirement 3: Comments explaining model choice based on dataset characteristics
    #
    # Dataset Characteristics:
    # - Samples: 178 (Small to medium-sized dataset)
    # - Features: 13 (Relatively low number of features)
    # - Classes: 3 (Multi-class classification)
    #
    # Model Choice Rationale:
    # Given the small to medium dataset size, a highly sophisticated model like SVC (Support Vector Classifier)
    # with a non-linear kernel (RBF) is an excellent choice.
    # 1.  SVC is powerful for classification tasks, capable of finding complex decision boundaries.
    # 2.  It performs well on datasets with a relatively low number of features and samples when properly scaled
    #     (which is handled by StandardScaler in the preprocessor).
    # 3.  Compared to MLPClassifier, SVC typically requires less data to train effectively and is less prone
    #     to overfitting on smaller datasets if hyperparameters are tuned correctly.
    # 4.  While GradientBoostingClassifier is powerful, it might be more prone to overfitting on very small datasets
    #     without extensive tuning. RandomForestClassifier is robust but SVC often offers superior performance
    #     on clean, smaller datasets like 'wine' when using an RBF kernel.
    # 5.  LogisticRegression is a good baseline but not considered "sophisticated" for this requirement.
    #
    # Hyperparameter Tuning for SVC:
    # - C=10: This regularization parameter balances misclassification of training examples and the simplicity
    #         of the decision surface. A value of 10 is chosen as a good balance, allowing for some flexibility
    #         without being too aggressive on the small dataset.
    # - gamma='scale': This parameter defines how far the influence of a single training example reaches.
    #                  'scale' uses 1 / (n_features * X.var()), which is a robust default that adapts to the data.
    # - kernel='rbf': The Radial Basis Function (RBF) kernel is chosen for its ability to model non-linear relationships.
    # - random_state=random_state: For reproducibility.
    model = SVC(C=10, gamma='scale', kernel='rbf', random_state=random_state)

    # Combine all steps into a final pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])