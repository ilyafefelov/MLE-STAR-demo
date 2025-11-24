# All Pipeline Wrapper Sources

Generated: 2025-11-19

This document contains the full source code of the pipeline wrapper files referenced by `configs/experiment_suite.yaml`. I copied these files verbatim so you can inspect them in one place.

---

## File: `model_comparison_results/gemini_live_california-housing-prices_pipeline_wrapper.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_full_pipeline(random_state=42):
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    feature_engineering = None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        PCA = None
    if PCA is not None:
        feature_engineering = Pipeline([("pca", PCA(n_components=0.95, random_state=random_state))])
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError('lightgbm not installed. Install it with: pip install lightgbm')
    model = lgb.LGBMRegressor(random_state=random_state)
    param_grid = None
    try:
        from sklearn.model_selection import GridSearchCV
    except Exception:
        GridSearchCV = None
    if GridSearchCV is not None:
        param_grid = {'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1]}
        model = GridSearchCV(model, param_grid=param_grid, cv=2, n_jobs=1)
    try:
        from sklearn.ensemble import VotingRegressor, VotingClassifier
    except Exception:
        VotingRegressor = None; VotingClassifier = None
    if VotingRegressor is not None or VotingClassifier is not None:
        from sklearn.linear_model import LinearRegression
        ensemble = VotingRegressor([("m1", model), ("m2", LinearRegression())])
        model = ensemble
    steps = [("preprocessor", preprocessor)]
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    return pipeline
```

---

## File: `model_comparison_results/gemini_2.5_flash_lite_digits_pipeline_wrapper.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
def build_full_pipeline(random_state: int = 42):
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

    # Feature Engineering: Dimensionality Reduction using PCA
    # Given the dataset size and number of features, PCA can help reduce
    # computational cost and potentially improve model performance by removing
    # redundant information. 64 features is a moderate number and can benefit
    # from dimensionality reduction without losing too much information.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=20, random_state=random_state)) # Chosen 20 components as a reasonable reduction
    ])

    # Model Selection and Hyperparameter Tuning
    # The digits dataset is a multi-class classification problem with a moderate
    # number of samples (1797) and features (64). Sophisticated models like
    # SVC, RandomForestClassifier, GradientBoostingClassifier, and MLPClassifier
    # are good candidates.
    #
    # - SVC: Often performs well on image-like datasets with well-defined decision boundaries.
    # - RandomForestClassifier: Robust and handles non-linearities well.
    # - GradientBoostingClassifier: Can achieve high accuracy by iteratively correcting errors.
    # - MLPClassifier: Powerful for complex patterns but can be sensitive to hyperparameters.
    #
    # Considering the need for a "BEST" and "SOPHISTICATED" model with tuned
    # hyperparameters for this dataset, SVC with a Radial Basis Function (RBF) kernel
    # is a strong contender due to its ability to handle complex, non-linear
    # relationships and its proven effectiveness on similar image classification tasks.
    # We'll tune its C and gamma parameters.

    # Define models and their parameter grids for GridSearchCV
    models_and_params = [
        ('SVC', SVC(random_state=random_state), {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf']
        }),
        ('RandomForestClassifier', RandomForestClassifier(random_state=random_state), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=random_state), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }),
        ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=500), {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        })
    ]

    best_model = None
    best_score = -1
    best_params = {}

    # To find the "BEST" model and tune hyperparameters, we would typically
    # run GridSearchCV on each model separately and then select the best.
    # For this function to return a single pipeline, we'll assume we've
    # pre-selected SVC as it's a common high-performer for this type of data
    # and provide its tuned parameters.

    # Example of how GridSearchCV would be used (but not executed here to return a static pipeline)
    #
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import load_digits
    #
    # X, y = load_digits(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    #
    # # Example for SVC
    # svc_model = SVC(random_state=random_state)
    # param_grid_svc = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    #     'kernel': ['rbf']
    # }
    # grid_search_svc = GridSearchCV(svc_model, param_grid_svc, cv=5, scoring='accuracy')
    # grid_search_svc.fit(preprocessor.fit_transform(X_train), y_train) # Note: This is simplified for illustration
    # print(f"Best params for SVC: {grid_search_svc.best_params_}")
    # print(f"Best score for SVC: {grid_search_svc.best_score_}")

    # Based on common benchmarks and tuning for the digits dataset,
    # SVC with the following parameters often yields excellent results.
    # These are representative tuned hyperparameters.
    model = SVC(C=10, gamma=0.01, kernel='rbf', random_state=random_state) # Chosen best model with tuned hyperparameters

    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])
```

---

## File: `model_comparison_results/gemini_2.5_flash_lite_breast_cancer_pipeline_wrapper.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    # Preprocessing: Imputation and Scaling
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Feature Engineering: Dimensionality Reduction
    # PCA is chosen because it's a common and effective technique for reducing the dimensionality
    # of datasets with a moderate number of features like this one (30 features).
    # It can help to remove noise and multicollinearity, potentially improving model performance
    # and reducing training time.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=random_state)) # Keep 95% of variance
    ])

    # Model Selection and Hyperparameter Tuning
    # Considering the dataset size (569 samples), number of features (30), and binary classification task,
    # several models can perform well.
    # - RandomForestClassifier: Generally robust, handles non-linearities, and less sensitive to feature scaling.
    # - SVC: Can be powerful with appropriate kernel and regularization, but can be sensitive to scaling.
    # - GradientBoostingClassifier: Powerful for complex relationships, but can be prone to overfitting if not tuned.
    # - MLPClassifier: Can learn complex patterns, but requires careful tuning and is sensitive to scaling.

    # We will use GridSearchCV to find the best performing model among these sophisticated options.
    # For demonstration purposes, let's focus on RandomForestClassifier as a strong candidate and tune it.
    # In a real-world scenario, you would create a meta-estimator or loop through different models
    # with their respective parameter grids.

    # RandomForestClassifier is often a good balance of performance and robustness for this type of dataset.
    # It's less prone to overfitting than Gradient Boosting with default settings and handles
    # the moderate number of samples and features well.

    model = RandomForestClassifier(random_state=random_state)

    # Hyperparameter tuning for RandomForestClassifier
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'bootstrap': [True, False]
    }

    # GridSearchCV will be applied to the *entire pipeline* later during model training.
    # For now, we define the model with its default or initial parameters and the grid for tuning.
    # The GridSearchCV object itself will be part of the final pipeline if we were to integrate it.
    # However, the requirement is to return a Pipeline where 'model' is the chosen model instance.
    # So, we'll assume the tuning happens *after* this function returns, by passing the returned
    # pipeline to GridSearchCV.

    # If we were to pre-tune and pass a specific instance:
    # Best parameters found via separate GridSearchCV execution (example):
    # best_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=random_state)
    # model = best_rf


    # Construct the full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])

    return full_pipeline
```

---

## File: `model_comparison_results/gemini_2.5_flash_lite_wine_pipeline_wrapper.py`

```python
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
```

---

## File: `model_comparison_results/gemini_2.5_flash_lite_iris_pipeline_wrapper.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
def build_full_pipeline(random_state: int = 42):
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
        ('pca', PCA(n_components=2, random_state=random_state))
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

    model_svc = SVC(random_state=random_state, probability=True) # probability=True is useful for some metrics
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
    # model_rf = RandomForestClassifier(random_state=random_state)
    # param_grid_rf = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5, 10]
    # }
    # grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    #
    # model_gb = GradientBoostingClassifier(random_state=random_state)
    # param_grid_gb = {
    #     'n_estimators': [50, 100, 150],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7]
    # }
    # grid_search_gb = GridSearchCV(model_gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
    #
    # model_mlp = MLPClassifier(random_state=random_state, max_iter=500, early_stopping=True, validation_fraction=0.1)
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
```

---

### Notes

- These files are verbatim copies of the pipeline wrapper modules used by the experiment manifest. If you want me to also include any additional wrappers present under `model_comparison_results/` (not referenced in the manifest), say so and I will append them.
- I intentionally did not modify the code; it's safe to inspect here. If you want, I can also generate a short summary for each wrapper (steps and estimator types) or extract the `build_full_pipeline()` return Pipeline step names programmatically.

---
