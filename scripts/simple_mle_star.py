#!/usr/bin/env python
"""
Simple MLE-STAR pipeline generation using Gemini API directly.
This bypasses ADK issues and generates a pipeline for iris dataset.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import google.generativeai as genai
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mle_star_ablation.prompts import PromptBuilder

def load_env(env_file: Path):
    if not env_file.exists():
        return
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k, v)

# Load .env if present, and map GEMINI_API_TOKEN to GOOGLE_API_KEY if present
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris', help='Dataset name (breast_cancer, wine, digits, iris)')
    parser.add_argument('--task-dir', type=str, default=None, help='Optional path to task dir with train/test files')
    parser.add_argument('--out-dir', type=str, default='generated_pipelines', help='Output directory for pipeline file')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to .env file')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Gemini model to use')
    return parser.parse_args()


args = parse_args()
load_env(Path(args.env_file))
if 'GEMINI_API_TOKEN' in os.environ and 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_TOKEN']

# Configure Gemini API
api_key = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

# Use the specified model
model = genai.GenerativeModel(args.model)

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

    # Create compact prompt for MLE-STAR style pipeline generation (reduce token footprint)
    builder = PromptBuilder(args.dataset)
    builder.add_role_context()
    builder.add_task_description(description)
    builder.add_data_sample(train_df)
    builder.add_requirements()
    builder.add_output_format()
    builder.add_constraints()
    
    prompt = builder.build()

    print("🤖 Sending request to Gemini API (deterministic generation: temperature=0, top_p=1)...")
    # Force deterministic generation if supported, fallback otherwise
    try:
        response = model.generate_content(prompt, temperature=0, top_p=1)
    except TypeError:
        response = model.generate_content(prompt)

    return response.text

def main():
    print("="*80)
    print("SIMPLE MLE-STAR PIPELINE GENERATION")
    print("="*80)

    # Set up paths
    # Default task dir if not provided
    if args.task_dir:
        task_dir = Path(args.task_dir)
    else:
        task_dir = Path("adk-samples/python/agents/machine-learning-engineering/machine_learning_engineering/tasks") / args.dataset
    output_dir = Path(args.out_dir)
    output_dir.mkdir(exist_ok=True)

    # Load task data
    print("📂 Loading task data...")
    description, train_df, test_df = load_task_data(task_dir)
    print(f"✅ Loaded data: {train_df.shape[0]} train samples, {test_df.shape[0]} test samples")

    # Generate pipeline
    print("🚀 Generating ML pipeline...")
    pipeline_code = generate_pipeline(description, train_df, test_df)

    # Save pipeline
    output_file = output_dir / f"pipeline_{args.dataset}_gemini_direct.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pipeline_code)

    print(f"✅ Pipeline saved to: {output_file}")

    # Save raw response for reproducibility and auditing
    raw_resp_file = output_dir / f"pipeline_{args.dataset}_raw_response.txt"
    with open(raw_resp_file, 'w', encoding='utf-8') as f:
        f.write(pipeline_code if isinstance(pipeline_code, str) else str(pipeline_code))
    # Save raw response and also attempt to extract python block + save a cleaned version
    cleaned_file = output_dir / f"pipeline_{args.dataset}_clean.py"
    # Save raw; now call cleanup script to extract python block
    import subprocess
    cleanup_cmd = [sys.executable, 'scripts/cleanup_generated_pipeline.py', '--src', str(output_file), '--out', str(cleaned_file)]
    try:
        subprocess.run(cleanup_cmd, check=True)
        print(f"✅ Cleaned pipeline saved to: {cleaned_file}")
    except subprocess.CalledProcessError:
        print('⚠️ Could not extract python block from generated response; cleaned file not written')

    print("="*80)
    print("PROCESS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()