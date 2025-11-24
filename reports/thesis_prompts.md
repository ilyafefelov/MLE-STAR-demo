# Representative Gemini prompt used for pipeline generation

The actual prompts that generated the pipeline code are not always captured verbatim in the repository. The generated Python files (e.g., `model_comparison_results/gemini_2.5_flash_lite_breast_cancer.py`) contain the final `build_full_pipeline()` function which reflects the model choice and hyperparameter decisions. Below is a representative prompt we used as input to Gemini to generate a pipeline for a classification dataset:

```
You are: an experienced ML engineer. Generate an sklearn Pipeline in Python for a classification dataset with the following constraints:
- Use standard data preprocessing (imputation, scaling) unless indicated.
- Consider PCA for feature engineering if it is beneficial.
- Include hyperparameter tuning via GridSearchCV where appropriate.
- Choose a robust classifier from {RandomForestClassifier, GradientBoostingClassifier, SVC, MLPClassifier} and explain why you chose it.
- Keep code modular, returning a `build_full_pipeline()` function that returns a fitted/unfitted sklearn `Pipeline` object.
- Use random_state=42 where applicable for reproducibility.
- Provide comments describing the reasons for each choice and tuned hyperparameters.
```

Notes:
- Different Gemini variants produced different code length, model choices and pipeline steps; these differences are recorded in `reports/table1_model_comparison.csv`.
- If you require the original prompt files recorded externally (e.g., in run logs), please ensure the Gemini environment records and exports the prompt (not all runs capture full prompts for privacy reasons).
