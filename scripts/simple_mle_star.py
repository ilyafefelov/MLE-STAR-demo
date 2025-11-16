#!/usr/bin/env python
"""
Simple MLE-STAR pipeline generation using Gemini API directly.
This bypasses ADK issues and generates a pipeline for iris dataset.
"""

import os
import pandas as pd
from pathlib import Path
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key='AIzaSyChxgm8aM4JHblbMz-152YoU6ULPjWvJg4')

# Use gemini-1.5-flash model (reliable and available)
model = genai.GenerativeModel('gemini-1.5-flash')

def load_task_data(task_dir: Path):
    """Load task data from directory."""
    desc_file = task_dir / "task_description.txt"
    train_file = task_dir / "train.csv"
    test_file = task_dir / "test.csv"

    with open(desc_file, 'r') as f:
        description = f.read()

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return description, train_df, test_df

def generate_pipeline(description: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Generate ML pipeline using Gemini."""

    # Create prompt for MLE-STAR style pipeline generation
    prompt = f"""
You are an expert machine learning engineer. Generate a complete, production-ready Python script for the following ML task.

Task Description:
{description}

Training Data Sample (first 5 rows):
{train_df.head().to_string()}

Training Data Shape: {train_df.shape}
Test Data Shape: {test_df.shape}

Features: {list(train_df.columns[:-1])}
Target: {train_df.columns[-1]}

Requirements:
1. Use scikit-learn for the pipeline
2. Include proper data preprocessing
3. Use appropriate ML algorithm for classification
4. Include cross-validation for evaluation
5. Generate predictions on test set
6. Save the pipeline and predictions
7. Include proper error handling and logging
8. Make the code modular and well-documented

Generate a complete Python script that:
- Loads the data from 'train.csv' and 'test.csv'
- Preprocesses the data appropriately
- Trains a model with cross-validation
- Makes predictions on test data
- Saves results to files

The script should be runnable as-is.
"""

    print("🤖 Sending request to Gemini API...")
    response = model.generate_content(prompt)

    return response.text

def main():
    print("="*80)
    print("SIMPLE MLE-STAR PIPELINE GENERATION")
    print("="*80)

    # Set up paths
    task_dir = Path("adk-samples/python/agents/machine-learning-engineering/machine_learning_engineering/tasks/iris")
    output_dir = Path("generated_pipelines")
    output_dir.mkdir(exist_ok=True)

    # Load task data
    print("📂 Loading task data...")
    description, train_df, test_df = load_task_data(task_dir)
    print(f"✅ Loaded data: {train_df.shape[0]} train samples, {test_df.shape[0]} test samples")

    # Generate pipeline
    print("🚀 Generating ML pipeline...")
    pipeline_code = generate_pipeline(description, train_df, test_df)

    # Save pipeline
    output_file = output_dir / "pipeline_iris_gemini_direct.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pipeline_code)

    print(f"✅ Pipeline saved to: {output_file}")

    # Try to run the generated pipeline
    print("▶️  Running generated pipeline...")
    try:
        exec(pipeline_code)
        print("✅ Pipeline executed successfully!")
    except Exception as e:
        print(f"❌ Error executing pipeline: {e}")
        print("Pipeline code saved but execution failed.")

    print("="*80)
    print("PROCESS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()