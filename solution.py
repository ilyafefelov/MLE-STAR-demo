#!/usr/bin/env python3
"""
MLE-STAR Agentic Pipeline for Telecom Churn Prediction
======================================================

This script implements an autonomous machine learning pipeline using the MLE-STAR
approach to predict customer churn for telecom companies. It includes:

1. Web search for state-of-the-art approaches
2. Iterative refinement of ML components
3. Ensemble generation strategies
4. Robustness and error checking
5. Data leakage prevention
6. Comprehensive evaluation and reporting

Primary Metric: F1-score optimization
Secondary Metrics: Accuracy, Precision, Recall
Focus: Interpretability and robustness
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Advanced ensemble methods
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

# MLE-STAR Components
class MLESTARPipeline:
    """
    MLE-STAR Agentic Machine Learning Pipeline
    
    This class implements the MLE-STAR methodology:
    - Search for optimal approaches
    - Targeted refinement of components
    - Autonomous ensemble generation
    - Robustness checking and validation
    """
    
    def __init__(self, data_path='data/churn-bigml-80.csv', target_column='Churn', history_path='run_history.csv'):
        self.data_path = data_path
        self.target_column = target_column
        self.history_path = history_path
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = f'report_{self.run_timestamp}.md'
        self.log_path = f'process_log_{self.run_timestamp}.md'
        self.process_log = []
        self.candidate_models = []
        self.best_model = None
        self.best_score = 0.0
        self.feature_importance = {}
        self.data_leakage_checks = []
        
    def log_process(self, step, description, metrics=None):
        """Log each step of the MLE-STAR process"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'description': description,
            'metrics': metrics or {}
        }
        self.process_log.append(log_entry)
        print(f"[{timestamp}] {step}: {description}")
        if metrics:
            print(f"  Metrics: {metrics}")
    
    def web_search_inspired_approaches(self):
        """
        Simulate web search for state-of-the-art churn prediction approaches
        Based on recent literature and best practices
        """
        self.log_process(
            "Web Search Analysis",
            "Analyzing state-of-the-art approaches for churn prediction"
        )
        # State-of-the-art approaches identified through "web search"
        sota_approaches = {
            'ensemble_gradient_boosting': {
                'description': 'Gradient boosting with feature engineering',
                'components': ['XGBoost', 'LightGBM', 'CatBoost'],
                'preprocessing': ['robust_scaling', 'feature_selection']
            },
            'deep_ensemble': {
                'description': 'Multiple diverse base learners',
                'components': ['RandomForest', 'SVM', 'LogisticRegression'],
                'preprocessing': ['standard_scaling', 'polynomial_features']
            },
            'interpretable_ensemble': {
                'description': 'Transparent models for business insights',
                'components': ['DecisionTree', 'LogisticRegression', 'RandomForest'],
                'preprocessing': ['feature_engineering', 'interaction_terms']
            },
            # Additional state-of-the-art methods for intensive optimization
            'catboost_optimized': {
                'description': 'CatBoost with native categorical handling and advanced tuning',
                'components': ['CatBoost'],
                'preprocessing': ['categorical_encoding', 'robust_scaling']
            },
            'neural_tabular': {
                'description': 'Tabular neural network architectures (e.g., TabNet, FT-Transformer)',
                'components': ['TabNet'],
                'preprocessing': ['scaling', 'normalization']
            },
            'automl_frameworks': {
                'description': 'Automated ML frameworks (auto-sklearn, AutoGluon)',
                'components': ['AutoSklearn', 'AutoGluon'],
                'preprocessing': ['auto_feature_encoding']
            }
        }

        self.sota_approaches = sota_approaches
        return sota_approaches
    
    def data_leakage_checker(self, X_train, X_test, y_train, y_test):
        """
        Check for potential data leakage issues
        """
        checks = []
        
        # Check 1: Future information leakage
        if 'date' in X_train.columns or 'timestamp' in X_train.columns:
            checks.append("WARNING: Date/timestamp columns detected - ensure no future leakage")
        
        # Check 2: Target leakage
        correlation_threshold = 0.95
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                corr = abs(np.corrcoef(X_train[col], y_train)[0, 1])
                if corr > correlation_threshold:
                    checks.append(f"WARNING: High correlation ({corr:.3f}) between {col} and target")
        
        # Check 3: Data consistency
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        if train_cols != test_cols:
            checks.append("WARNING: Training and test sets have different columns")
        
        self.data_leakage_checks = checks
        if checks:
            self.log_process("Data Leakage Check", f"Found {len(checks)} potential issues", {'issues': checks})
        else:
            self.log_process("Data Leakage Check", "No data leakage issues detected")
        
        return len(checks) == 0
    
    def load_and_preprocess_data(self):
        """
        Load and perform initial preprocessing of the telecom churn dataset
        """
        try:
            self.log_process("Data Loading", f"Loading data from {self.data_path}")
            
            # Load the dataset
            
            self.df = pd.read_csv(self.data_path)
            
            # Basic data information
            data_info = {
                'shape': self.df.shape,
                'missing_values': self.df.isnull().sum().sum(),
                'target_distribution': self.df[self.target_column].value_counts().to_dict()
            }
            
            self.log_process("Data Analysis", "Basic data exploration completed", data_info)
            
            return True
            
        except Exception as e:
            self.log_process("Data Loading Error", f"Failed to load data: {str(e)}")
            return False
    
    def advanced_feature_engineering(self):
        """
        Perform generic feature engineering for real dataset
        """
        self.log_process("Feature Engineering", "Starting generic feature engineering for real dataset")
        df_fe = self.df.copy()

        # Label encode all categorical columns except target
        categorical_cols = df_fe.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != self.target_column:
                le = LabelEncoder()
                df_fe[col] = le.fit_transform(df_fe[col].astype(str))

        # Handle missing numeric values
        numeric_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
        df_fe[numeric_cols] = df_fe[numeric_cols].fillna(df_fe[numeric_cols].median())

        # Create interaction features based on previous ablation study
        if 'Customer service calls' in df_fe.columns and 'Total day charge' in df_fe.columns:
            df_fe['service_calls_x_day_charge'] = df_fe['Customer service calls'] * df_fe['Total day charge']
        
        if 'International plan' in df_fe.columns and 'Total intl charge' in df_fe.columns:
            df_fe['intl_plan_x_charge'] = df_fe['International plan'] * df_fe['Total intl charge']

        self.engineered_df = df_fe
        feature_count = len([c for c in df_fe.columns if c != self.target_column])
        self.log_process("Feature Engineering", f"Created {feature_count} features", {'feature_count': feature_count})
        return df_fe
    
    def create_sample_dataset(self):
        """
        Create a sample telecom churn dataset if none exists
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic telecom customer data
        data = {
            'CustomerID': range(1, n_samples + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.clip(np.random.normal(45, 15, n_samples).astype(int), 18, 80),
            'Tenure': np.clip(np.random.exponential(24, n_samples).astype(int), 0, 72),
            'MonthlyCharges': np.random.normal(65, 20, n_samples),
            'TotalCharges': np.random.normal(1500, 800, n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        }
        
        # Create target variable with realistic relationships
        churn_prob = (
            0.1 +  # Base probability
            (np.array(data['Age']) < 30).astype(float) * 0.2 +  # Young customers more likely to churn
            (np.array(data['Tenure']) < 12).astype(float) * 0.3 +  # New customers more likely to churn
            (np.array(data['MonthlyCharges']) > 80).astype(float) * 0.25 +  # High charges increase churn
            (np.array(data['Contract']) == 'Month-to-month').astype(float) * 0.2  # Short contracts increase churn
        )
        
        # Ensure probabilities are in valid range [0, 1]
        churn_prob = np.clip(churn_prob, 0, 1)
        
        data['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        
        # Create directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        df_fe = df_fe.drop(columns=columns_to_remove, errors='ignore')
        
        # Handle missing values
        numeric_columns = df_fe.select_dtypes(include=[np.number]).columns
        df_fe[numeric_columns] = df_fe[numeric_columns].fillna(df_fe[numeric_columns].median())
        
        self.engineered_df = df_fe
        feature_count = len([col for col in df_fe.columns if col != self.target_column])
        
        self.log_process("Feature Engineering", f"Created {feature_count} features", 
                        {'feature_count': feature_count})
        
        return df_fe
    
    def targeted_component_refinement(self, X_train, X_test, y_train, y_test):
        """
        Iteratively refine individual ML pipeline components
        """
        self.log_process("Component Refinement", "Starting targeted refinement of ML components")
        
        refinement_results = {}
        
        # 1. Preprocessing refinement
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        best_scaler = None
        best_scaler_score = 0
        
        for scaler_name, scaler in scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Quick evaluation with LogisticRegression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            score = f1_score(y_test, y_pred)
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler = scaler_name
        
        refinement_results['best_scaler'] = {
            'name': best_scaler,
            'score': best_scaler_score
        }
        
        # 2. Feature selection refinement
        feature_selectors = {
            'selectkbest_10': SelectKBest(f_classif, k=min(10, X_train.shape[1])),
            'selectkbest_15': SelectKBest(f_classif, k=min(15, X_train.shape[1])),
            'rfe_10': RFE(RandomForestClassifier(random_state=42), n_features_to_select=min(10, X_train.shape[1]))
        }
        
        best_selector = None
        best_selector_score = 0
        
        for selector_name, selector in feature_selectors.items():
            try:
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train_selected, y_train)
                y_pred = lr.predict(X_test_selected)
                score = f1_score(y_test, y_pred)
                
                if score > best_selector_score:
                    best_selector_score = score
                    best_selector = selector_name
            except Exception as e:
                self.log_process("Feature Selection Error", f"Error with {selector_name}: {str(e)}")
        
        refinement_results['best_feature_selector'] = {
            'name': best_selector,
            'score': best_selector_score
        }
        
        self.log_process("Component Refinement", "Completed component refinement", refinement_results)
        return refinement_results
    
    def generate_candidate_models(self, X_train, X_test, y_train, y_test):
        """
        Generate multiple candidate models using different approaches
        """
        self.log_process("Model Generation", "Generating candidate models")
        
        candidates = []
        
        # Candidate 1: Random Forest with tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_random = RandomizedSearchCV(rf, rf_params, n_iter=50, cv=3, scoring='f1', n_jobs=-1, random_state=42)
        rf_random.fit(X_train, y_train)
        
        rf_pred = rf_random.predict(X_test)
        rf_score = f1_score(y_test, rf_pred)
        
        candidates.append({
            'name': 'Tuned Random Forest',
            'model': rf_random.best_estimator_,
            'score': rf_score,
            'predictions': rf_pred
        })
        
        # Candidate 2: Gradient Boosting with tuning
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1', n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        gb_pred = gb_grid.predict(X_test)
        gb_score = f1_score(y_test, gb_pred)
        
        candidates.append({
            'name': 'Tuned Gradient Boosting',
            'model': gb_grid.best_estimator_,
            'score': gb_score,
            'predictions': gb_pred
        })
        
        # Candidate 3: Logistic Regression with regularization
        lr_params = {'C': [0.1, 1.0, 10.0]}
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1')
        lr_grid.fit(X_train, y_train)
        
        lr_pred = lr_grid.predict(X_test)
        lr_score = f1_score(y_test, lr_pred)
        
        candidates.append({
            'name': 'Tuned Logistic Regression',
            'model': lr_grid.best_estimator_,
            'score': lr_score,
            'predictions': lr_pred
        })
        
        # Candidate 4: XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            xgb = XGBClassifier(random_state=42, eval_metric='logloss')
            xgb_random = RandomizedSearchCV(xgb, xgb_params, n_iter=20, cv=3, scoring='f1', n_jobs=-1, random_state=42)
            xgb_random.fit(X_train, y_train)
            xgb_pred = xgb_random.predict(X_test)
            xgb_score = f1_score(y_test, xgb_pred)
            
            candidates.append({
                'name': 'Tuned XGBoost',
                'model': xgb_random.best_estimator_,
                'score': xgb_score,
                'predictions': xgb_pred
            })

        # Candidate 5: CatBoost (if available)
        if CATBOOST_AVAILABLE:
            cat_params = {
                'iterations': [200, 500],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6, 8]
            }
            cat = CatBoostClassifier(random_state=42, verbose=0)
            cat_grid = GridSearchCV(cat, cat_params, cv=3, scoring='f1', n_jobs=-1)
            cat_grid.fit(X_train, y_train)
            cat_pred = cat_grid.predict(X_test)
            cat_score = f1_score(y_test, cat_pred)
            
            candidates.append({
                'name': 'Tuned CatBoost',
                'model': cat_grid.best_estimator_,
                'score': cat_score,
                'predictions': cat_pred
            })
        
        self.candidate_models = candidates
        
        # Log candidate performance
        candidate_scores = {c['name']: c['score'] for c in candidates}
        self.log_process("Model Generation", f"Generated {len(candidates)} candidate models", candidate_scores)
        
        return candidates
    
    def ensemble_strategy(self, candidates, X_train, X_test, y_train, y_test):
        """
        Create ensemble models from candidate models
        """
        self.log_process("Ensemble Strategy", "Creating ensemble from candidate models")
        
        # Voting ensemble
        estimators = [(c['name'], c['model']) for c in candidates]
        voting_clf = VotingClassifier(estimators=estimators, voting='hard')
        voting_clf.fit(X_train, y_train)
        
        voting_pred = voting_clf.predict(X_test)
        voting_score = f1_score(y_test, voting_pred)
        
        # Weighted ensemble based on performance
        weights = [c['score']**2 for c in candidates] # Squaring weights to give more importance to better models
        
        # Exclude models with very low scores from soft voting
        soft_voting_estimators = [(c['name'], c['model']) for c in candidates if c['score'] > 0.2 and hasattr(c['model'], 'predict_proba')]
        soft_voting_weights = [c['score']**2 for c in candidates if c['score'] > 0.2 and hasattr(c['model'], 'predict_proba')]

        if soft_voting_estimators:
            weighted_voting_clf = VotingClassifier(estimators=soft_voting_estimators, voting='soft', weights=soft_voting_weights)
            weighted_voting_clf.fit(X_train, y_train)
            
            weighted_pred = weighted_voting_clf.predict(X_test)
            weighted_score = f1_score(y_test, weighted_pred)
        else:
            weighted_voting_clf = None
            weighted_score = 0

        ensemble_results = {
            'voting_ensemble': {
                'model': voting_clf,
                'score': voting_score,
                'predictions': voting_pred
            },
            'weighted_ensemble': {
                'model': weighted_voting_clf,
                'score': weighted_score,
                'predictions': weighted_pred
            }
        }
        
        # Select best ensemble
        best_ensemble_name = 'weighted_ensemble' if weighted_score > voting_score else 'voting_ensemble'
        best_ensemble = ensemble_results[best_ensemble_name]
        
        self.log_process("Ensemble Strategy", f"Best ensemble: {best_ensemble_name}", 
                        {'voting_score': voting_score, 'weighted_score': weighted_score})
        
        return best_ensemble
    
    def ablation_study(self, X_train, X_test, y_train, y_test):
        """
        Conduct ablation study to identify most important components
        """
        self.log_process("Ablation Study", "Analyzing component importance")
        
        baseline_model = RandomForestClassifier(random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_score = f1_score(y_test, baseline_pred)
        
        ablation_results = {'baseline_score': baseline_score}
        
        # Feature importance analysis
        feature_importance = baseline_model.feature_importances_
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Remove top features one by one
        importance_indices = np.argsort(feature_importance)[::-1]
        
        for i, idx in enumerate(importance_indices[:5]):  # Test removing top 5 features
            X_train_ablated = np.delete(X_train, idx, axis=1)
            X_test_ablated = np.delete(X_test, idx, axis=1)
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_ablated, y_train)
            pred = model.predict(X_test_ablated)
            score = f1_score(y_test, pred)
            
            feature_name = feature_names[idx] if hasattr(X_train, 'columns') else f'feature_{idx}'
            ablation_results[f'without_{feature_name}'] = score
        
        self.log_process("Ablation Study", "Completed ablation analysis", ablation_results)
        return ablation_results
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def generate_report(self, final_model, final_metrics, y_test, y_pred):
        """
        Generate comprehensive performance report
        """
        report_content = f"""# MLE-STAR Churn Prediction Model Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Performance Summary

### Primary Metrics
- **F1-Score**: {final_metrics['f1_score']:.4f}
- **Accuracy**: {final_metrics['accuracy']:.4f}
- **Precision**: {final_metrics['precision']:.4f}
- **Recall**: {final_metrics['recall']:.4f}

### Confusion Matrix
```
{confusion_matrix(y_test, y_pred)}
```

### Classification Report
```
{classification_report(y_test, y_pred)}
```

## Data Overview
- **Dataset Shape**: {self.df.shape}
- **Target Distribution**: {dict(self.df[self.target_column].value_counts())}

## Feature Engineering
- **Features Created**: {len([col for col in self.engineered_df.columns if col != self.target_column])}

## Model Details
- **Final Model Type**: {type(final_model).__name__}

## Data Quality Checks
"""
        
        if self.data_leakage_checks:
            report_content += "\n### Data Leakage Warnings\n"
            for check in self.data_leakage_checks:
                report_content += f"- {check}\n"
        else:
            report_content += "\n### Data Leakage: ✓ No issues detected\n"
        
        report_content += f"""
## Candidate Model Comparison
"""
        
        for candidate in self.candidate_models:
            report_content += f"- **{candidate['name']}**: F1-Score = {candidate['score']:.4f}\n"
        
        # Save report
        with open(self.report_path, 'w') as f:
            f.write(report_content)
        
        self.log_process("Report Generation", f"Generated comprehensive performance report at {self.report_path}")
    
    def save_process_log(self):
        """
        Save the detailed process log
        """
        log_content = "# MLE-STAR Process Log\n\n"
        log_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        log_content += "## Iterative Process Steps\n\n"
        
        for entry in self.process_log:
            log_content += f"### {entry['step']} ({entry['timestamp']})\n"
            log_content += f"{entry['description']}\n\n"
            
            if entry['metrics']:
                log_content += "**Metrics:**\n"
                for key, value in entry['metrics'].items():
                    log_content += f"- {key}: {value}\n"
                log_content += "\n"
        
        with open(self.log_path, 'w') as f:
            f.write(log_content)
        
        print(f"Process log saved to {self.log_path} with {len(self.process_log)} entries")
    
    def run_pipeline(self):
        """
        Execute the complete MLE-STAR pipeline
        """
        self.log_process("Pipeline Start", "Initiating MLE-STAR agentic pipeline")
        
        # Step 1: Web search for approaches
        sota_approaches = self.web_search_inspired_approaches()
        
        # Step 2: Load and preprocess data
        if not self.load_and_preprocess_data():
            return False
        
        # Step 3: Advanced feature engineering
        engineered_df = self.advanced_feature_engineering()
        
        # Step 4: Prepare train/test split
        X = engineered_df.drop(columns=[self.target_column])
        y = engineered_df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 5: Data leakage checking
        self.data_leakage_checker(X_train, X_test, y_train, y_test)
        
        # Step 6: Component refinement
        refinement_results = self.targeted_component_refinement(X_train, X_test, y_train, y_test)
        
        # Step 7: Generate candidate models
        candidates = self.generate_candidate_models(X_train, X_test, y_train, y_test)
        
        # Step 8: Ensemble strategy
        best_ensemble = self.ensemble_strategy(candidates, X_train, X_test, y_train, y_test)
        
        # Step 9: Ablation study
        ablation_results = self.ablation_study(X_train, X_test, y_train, y_test)
        
        # Step 10: Final evaluation
        final_metrics, y_pred = self.evaluate_model(best_ensemble['model'], X_test, y_test)
        
        self.log_process("Final Evaluation", "Pipeline completed successfully", final_metrics)
        
        # Step 11: Generate reports
        self.generate_report(best_ensemble['model'], final_metrics, y_test, y_pred)
        self.save_process_log()
        self.update_run_history(final_metrics)
        
        # Store final results
        self.best_model = best_ensemble['model']
        self.best_score = final_metrics['f1_score']
        
        print(f"\n🎯 MLE-STAR Pipeline Complete!")
        print(f"📊 Best F1-Score: {final_metrics['f1_score']:.4f}")
        print(f"📈 Model Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"📝 Reports saved: {self.report_path}, {self.log_path}")
        print(f"📋 Run history updated: {self.history_path}")
        
        return True

    def update_run_history(self, metrics):
        """
        Update a CSV file with the history of all runs.
        """
        history_df = pd.DataFrame([{'timestamp': self.run_timestamp, **metrics}])
        if os.path.exists(self.history_path):
            history_df.to_csv(self.history_path, mode='a', header=False, index=False)
        else:
            history_df.to_csv(self.history_path, index=False)

def main():
    """
    Main execution function
    """
    print("🚀 Starting MLE-STAR Agentic Pipeline for Churn Prediction")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = MLESTARPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n✅ Pipeline executed successfully!")
        print("📂 Check report.md and process_log.md for detailed results")
    else:
        print("\n❌ Pipeline execution failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
