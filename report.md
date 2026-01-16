# MLE-STAR Churn Prediction Model Report

Generated on: 2025-08-04 10:41:18

## Model Performance Summary

### Primary Metrics
- **F1-Score**: 0.8406
- **Accuracy**: 0.9588
- **Precision**: 0.9667
- **Recall**: 0.7436

### Confusion Matrix
```
[[454   2]
 [ 20  58]]
```

### Classification Report
```
              precision    recall  f1-score   support

       False       0.96      1.00      0.98       456
        True       0.97      0.74      0.84        78

    accuracy                           0.96       534
   macro avg       0.96      0.87      0.91       534
weighted avg       0.96      0.96      0.96       534

```

## Data Overview
- **Dataset Shape**: (2666, 20)
- **Target Distribution**: {False: np.int64(2278), True: np.int64(388)}

## Feature Engineering
- **Features Created**: 21

## Model Details
- **Final Model Type**: VotingClassifier

## Data Quality Checks

### Data Leakage: ✓ No issues detected

## Candidate Model Comparison
- **Tuned Random Forest**: F1-Score = 0.8148
- **Tuned Gradient Boosting**: F1-Score = 0.7917
- **Tuned Logistic Regression**: F1-Score = 0.4480
- **Tuned XGBoost**: F1-Score = 0.8201
- **Tuned CatBoost**: F1-Score = 0.8321
