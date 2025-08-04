# MLE-STAR Churn Prediction Model Report

Generated on: 2025-08-04 09:55:49

## Model Performance Summary

### Primary Metrics
- **F1-Score**: 0.4889
- **Accuracy**: 0.6550
- **Precision**: 0.5593
- **Recall**: 0.4342

### Confusion Matrix
```
[[98 26]
 [43 33]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       0.70      0.79      0.74       124
           1       0.56      0.43      0.49        76

    accuracy                           0.66       200
   macro avg       0.63      0.61      0.61       200
weighted avg       0.64      0.66      0.64       200

```

## Data Overview
- **Dataset Shape**: (1000, 11)
- **Target Distribution**: {0: np.int64(622), 1: np.int64(378)}

## Feature Engineering
- **Features Created**: 12

## Model Details
- **Final Model Type**: VotingClassifier

## Data Quality Checks

### Data Leakage: ✓ No issues detected

## Candidate Model Comparison
- **Tuned Random Forest**: F1-Score = 0.4812
- **Gradient Boosting**: F1-Score = 0.4511
- **Tuned Logistic Regression**: F1-Score = 0.4375
- **XGBoost**: F1-Score = 0.5278
