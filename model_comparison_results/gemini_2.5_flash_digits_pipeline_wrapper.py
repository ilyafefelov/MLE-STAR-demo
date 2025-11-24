from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
def build_full_pipeline(random_state: int = 42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    # No need to import other models as MLPClassifier was chosen

    # Preprocessing step: Imputation (strategy='mean') and Scaling (StandardScaler)
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Feature Engineering step: Dimensionality reduction using PCA
    # For the 'digits' dataset (image-like data), PCA is an effective method to:
    # 1. Reduce the number of features (from 64) while retaining most of the variance.
    # 2. Extract salient features and potentially reduce noise, leading to better model performance.
    # n_components=0.95 aims to capture 95% of the variance, providing a good balance
    # between dimensionality reduction and information retention.
    feature_engineering = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=random_state))
    ])

    # Model choice: MLPClassifier (Multi-layer Perceptron Classifier - a Neural Network)
    # Rationale for MLPClassifier:
    # - Dataset size (1797 samples): Medium-small. MLPClassifier can effectively learn complex
    #   non-linear patterns from this amount of data without being excessively slow, especially
    #   when combined with efficient solvers like 'adam'.
    # - Features (64 reduced by PCA): A neural network excels at finding intricate relationships
    #   within feature vectors, even after dimensionality reduction, which is crucial for
    #   distinguishing between digits based on pixel intensities.
    # - Classes (10): MLPClassifier inherently supports multi-class classification and is well-suited
    #   for tasks with multiple distinct categories like digit recognition.
    # - Sophistication: MLPClassifier is a powerful, non-linear model. It represents a sophisticated
    #   choice beyond linear models or simpler tree ensembles, often achieving high accuracy on
    #   image-like classification tasks.
    #
    # Hyperparameter Tuning for the 'digits' dataset:
    # - hidden_layer_sizes=(100, 50): Defines two hidden layers with 100 and 50 neurons respectively.
    #   This provides sufficient capacity for learning the complexity of digit features without
    #   being overly deep for the dataset size, helping to capture hierarchical features.
    # - activation='relu': Rectified Linear Unit, a standard and effective activation function
    #   for hidden layers, promoting sparsity and faster convergence.
    # - solver='adam': An efficient stochastic gradient-based optimizer that generally performs
    #   well across various datasets and network architectures, requiring less manual learning
    #   rate tuning than 'sgd'.
    # - alpha=0.0001: L2 regularization term. A small value helps prevent overfitting by penalizing
    #   large weights.
    # - max_iter=300: The maximum number of epochs. This value provides enough iterations for
    #   the network to converge on many datasets, balancing training time and performance.
    # - random_state=random_state: Ensures reproducibility of the results.
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=300,
        random_state=random_state,
        early_stopping=True, # Added for better generalization
        validation_fraction=0.1 # Used with early_stopping
    )

    # Combine all steps into a final scikit-learn Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])