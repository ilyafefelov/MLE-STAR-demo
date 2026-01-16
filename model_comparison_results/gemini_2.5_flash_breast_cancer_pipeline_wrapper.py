from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.svm import SVC
    # from sklearn.neural_network import MLPClassifier

    # Step 1: Preprocessing
    # Impute missing values (if any) with the mean, then scale features to standard normal distribution.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Step 2: Feature Engineering (Dimensionality Reduction)
    # For a dataset with 30 features and 569 samples, PCA can effectively reduce dimensionality
    # while retaining most of the variance. This can help to reduce noise, prevent overfitting
    # for subsequent models, and potentially speed up training.
    # We'll reduce the number of features from 30 to 10 components.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=10, random_state=random_state))
    ])

    # Step 3: Model Selection and Tuning
    # Dataset Characteristics:
    # - Samples: 569 (considered small to medium-sized)
    # - Features: 30 (moderate number)
    # - Classes: 2 (binary classification)
    #
    # Model Choice: GradientBoostingClassifier
    # Justification:
    # 1. Sophistication: Gradient Boosting is a powerful ensemble technique known for achieving
    #    high predictive accuracy on tabular datasets, often outperforming simpler models like
    #    Logistic Regression and sometimes even RandomForest in terms of raw performance.
    # 2. Dataset Size (569 samples): While a deep neural network (MLP) might struggle with
    #    generalization on such a relatively small dataset without extensive tuning and validation,
    #    Gradient Boosting is robust and can learn complex patterns without being excessively prone
    #    to overfitting when combined with appropriate hyperparameters and potentially dimensionality
    #    reduction (like PCA). SVC with an RBF kernel can also be powerful but can be slower and
    #    more sensitive to scaling and hyperparameters for this scale.
    # 3. Features (30 features): It handles a moderate number of features well, and benefits from
    #    the prior PCA step which extracts the most informative components, potentially reducing
    #    collinearity and noise.
    # 4. Binary Classification: Gradient Boosting is well-suited for binary classification tasks.
    #
    # Hyperparameter Tuning for Breast Cancer Dataset:
    # The chosen hyperparameters are tuned to provide a good balance between performance and
    # preventing overfitting for this specific dataset size and characteristics.
    # - n_estimators: A moderate number of boosting stages. Too many can overfit, too few underfit.
    # - learning_rate: Controls the contribution of each tree. Smaller values require more estimators
    #   but lead to a more robust model.
    # - max_depth: Limits the depth of individual trees. Smaller depths prevent individual trees
    #   from overfitting the training data too much, common in boosting.
    # - subsample: Fraction of samples used for fitting the individual base learners. Can reduce
    #   variance and overfitting.
    # - random_state: Ensures reproducibility of the results.
    model = GradientBoostingClassifier(
        n_estimators=150,      # Number of boosting stages
        learning_rate=0.07,    # Step size shrinkage to prevent overfitting
        max_depth=4,           # Maximum depth of the individual regression estimators
        subsample=0.8,         # Fraction of samples used for fitting the individual base learners
        random_state=random_state
    )

    # Combine all steps into a single scikit-learn Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])