# MLE-STAR Churn Prediction Model Report

Generated on: 2025-08-04 10:34:31

## Model Performance Summary

### Primary Metrics
- **F1-Score**: 0.8382
- **Accuracy**: 0.9588
- **Precision**: 0.9828
- **Recall**: 0.7308

### Confusion Matrix
```
[[455   1]
 [ 21  57]]
```

### Classification Report
```
              precision    recall  f1-score   support

       False       0.96      1.00      0.98       456
        True       0.98      0.73      0.84        78

    accuracy                           0.96       534
   macro avg       0.97      0.86      0.91       534
weighted avg       0.96      0.96      0.96       534

```

## Data Overview
- **Dataset Shape**: (2666, 20)
- **Target Distribution**: {False: np.int64(2278), True: np.int64(388)}

## Feature Engineering
- **Features Created**: 19

## Model Details
- **Final Model Type**: VotingClassifier

## Data Quality Checks

### Data Leakage: ✓ No issues detected

## Candidate Model Comparison
- **Tuned Random Forest**: F1-Score = 0.8120
- **Gradient Boosting**: F1-Score = 0.7910
- **Tuned Logistic Regression**: F1-Score = 0.1942
- **XGBoost**: F1-Score = 0.8286
- **CatBoost**: F1-Score = 0.8382
