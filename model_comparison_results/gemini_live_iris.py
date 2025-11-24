
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
# Assuming train.csv and test.csv are available in the same directory.
# If not, provide the correct path to the files.
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    # For demonstration purposes, let's load the Iris dataset directly if files are not found
    # In a real scenario, you would ensure the files are present.
    from sklearn.datasets import load_iris
    iris = load_iris()
    train_df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
    # Create a dummy test set by splitting the loaded iris dataset
    X_iris, y_iris = iris.data, iris.target
    _, X_test_iris, _, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)
    test_df_iris = pd.DataFrame(data=X_test_iris, columns=iris.feature_names)
    test_df = test_df_iris # Use this for prediction if train.csv was not found
    train_df['target'] = train_df['target'].astype(int) # Ensure target is integer

# Separate target variable
# Check if 'target' column exists, if not, it might be the Titanic dataset and needs different handling.
# Based on the task description, we expect 'target' for Iris.
if 'target' in train_df.columns:
    X = train_df.drop('target', axis=1)
    y = train_df['target']
else:
    # This block would handle the Titanic dataset if 'target' is not found.
    # However, the task is for Iris, so we raise an error or handle accordingly.
    raise ValueError("The 'target' column was not found in train.csv. Please ensure you are using the Iris dataset.")

# --- Preprocessing ---
# For the Iris dataset, the features are already numerical and clean.
# No specific preprocessing like imputation or encoding is needed for these features.
# We will use the features directly.

# Define feature columns based on train_df (excluding target if it was there)
feature_cols = [col for col in train_df.columns if col != 'target']

# Ensure test_df has the same feature columns as train_df
# This step is crucial if train.csv and test.csv were actually different files
# and test.csv might be missing columns or have extra ones.
# For Iris, this is less of an issue as features are standard.
X_test = test_df[feature_cols]


# --- Model Training and Hyperparameter Tuning ---

# Define the model
hgb_model = HistGradientBoostingClassifier(random_state=42)

# Define a parameter grid for hyperparameter tuning
# These are common parameters for gradient boosting models.
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'l2_regularization': [0.0, 0.1, 0.5, 1.0]
}

# Set up GridSearchCV for hyperparameter tuning
# Using 5-fold cross-validation
grid_search = GridSearchCV(estimator=hgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X, y)

# Get the best model and best parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
final_validation_score = grid_search.best_score_

print(f"Final Validation Performance: {final_validation_score:.4f}")
print(f"Best Hyperparameters: {best_params}")

# --- Model Evaluation ---

# Predict on the test data using the best model
y_pred_test = best_model.predict(X_test)

# Calculate accuracy on the test set (if test labels were available and loaded)
# For this task, we only need to generate predictions for submission.
# If y_test_iris was loaded, you could uncomment the following lines:
# test_accuracy = accuracy_score(y_test_iris, y_pred_test)
# print(f"Test Set Accuracy: {test_accuracy:.4f}")

# --- Submission File Generation ---

# Create submission DataFrame
# If the original test_df was loaded from a file, it would contain 'PassengerId' or similar.
# For the Iris dataset loaded directly, we'll create a dummy index for submission.
submission_df = pd.DataFrame({'target': y_pred_test})

# Save submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
