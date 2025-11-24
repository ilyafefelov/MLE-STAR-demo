from typing import Optional, List, Dict, Any

class PromptBuilder:
    """
    A modular builder for constructing prompts for the MLE-STAR agent.
    Allows assembling the prompt from distinct sections (Role, Task, Context, Constraints).
    """
    def __init__(self, dataset_name: str, dataset_info: Optional[Dict] = None):
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info or {}
        self.sections = []

    def add_role_context(self):
        """Adds the persona/role definition."""
        self.sections.append("You are an expert Machine Learning Engineer.")
        return self

    def add_task_description(self, description: Optional[str] = None):
        """Adds the high-level task description."""
        if description:
            self.sections.append(f"Task Description:\n{description}")
        else:
            self.sections.append(f"Generate a complete scikit-learn ML pipeline for the '{self.dataset_name}' dataset.")
        return self

    def add_dataset_info(self):
        """Adds specific information about the dataset."""
        if self.dataset_info:
            info_str = "Dataset Information:\n"
            if 'n_samples' in self.dataset_info:
                info_str += f"- Samples: {self.dataset_info['n_samples']}\n"
            if 'n_features' in self.dataset_info:
                info_str += f"- Features: {self.dataset_info['n_features']}\n"
            if 'n_classes' in self.dataset_info:
                info_str += f"- Classes: {self.dataset_info['n_classes']}\n"
            self.sections.append(info_str)
        return self

    def add_data_sample(self, train_df: Any):
        """Adds a sample of the training data."""
        if train_df is not None:
            sample_str = f"""
Training Data Sample (first 3 rows):
{train_df.head(3).to_string()}

Training Data Shape: {train_df.shape}
Features: {list(train_df.columns[:-1])}
Target: {train_df.columns[-1]}
"""
            self.sections.append(sample_str)
        return self

    def add_requirements(self):
        """Adds the technical requirements for the pipeline."""
        reqs = """Requirements:
1. Create a Pipeline with these steps:
   - 'preprocessor': SimpleImputer + StandardScaler
   - 'feature_engineering': Dimensionality reduction or feature extraction
   - 'model': Choose BEST model from: LogisticRegression, RandomForestClassifier, 
     SVC, GradientBoostingClassifier, MLPClassifier

2. Return ONLY Python function code with EXACT signature:
    `def build_full_pipeline(random_state: int = 42, numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None) -> Pipeline:`
"""
        self.sections.append(reqs)
        return self

    def add_output_format(self):
        """Adds the expected code structure/template."""
        fmt = """
```python
def build_full_pipeline(random_state: int = 42, numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None) -> Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    # ... imports
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    feature_engineering = Pipeline([
        # your choice based on dataset
    ])
    
    model = # YOUR BEST MODEL CHOICE with tuned hyperparameters
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineering', feature_engineering),
        ('model', model)
    ])
```
"""
        self.sections.append(fmt)
        return self

    def add_constraints(self):
        """Adds constraints and specific instructions for the model choice."""
        constraints = f"""
3. Add comments explaining model choice based on:
   - Dataset size ({self.dataset_info.get('n_samples', 'N/A')} samples)
   - Features ({self.dataset_info.get('n_features', 'N/A')})
   - Classes ({self.dataset_info.get('n_classes', 'N/A')})

4. Use random_state=42
5. Choose SOPHISTICATED model - not just LogisticRegression!
6. Tune hyperparameters for this specific dataset
7. ENSURE the pipeline includes at least two of the following components: 'preprocessor' (impute + scaling), 'feature_engineering' (PCA/Poly/SelectKBest), and 'tuning' (GridSearch or RandomizedSearch). This is to ensure ablation has measurable effects.

Generate ONLY code, no explanations outside.
"""
        self.sections.append(constraints)
        return self

    def build(self) -> str:
        """Assembles the final prompt string."""
        return "\n\n".join(self.sections)

def generate_mle_prompt(dataset_name: str, dataset_info: Dict) -> str:
    """
    Helper function to generate the standard MLE prompt using the builder.
    """
    builder = PromptBuilder(dataset_name, dataset_info)
    return (builder
            .add_role_context()
            .add_task_description()
            .add_dataset_info()
            .add_requirements()
            .add_output_format()
            .add_constraints()
            .build())
