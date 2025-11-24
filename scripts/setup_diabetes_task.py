import os
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define path
task_dir = Path('adk-samples/python/agents/machine-learning-engineering/machine_learning_engineering/tasks/diabetes')
task_dir.mkdir(parents=True, exist_ok=True)

# Load data
data = load_diabetes(as_frame=True)
df = data.frame

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save
train_df.to_csv(task_dir / 'train.csv', index=False)
test_df.to_csv(task_dir / 'test.csv', index=False)

print(f"Created diabetes dataset in {task_dir}")
