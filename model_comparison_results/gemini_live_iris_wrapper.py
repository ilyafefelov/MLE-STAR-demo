"""
Adapter wrapper to use a generated pipeline (pipeline_iris_clean.py) in the ablation harness.

defines build_full_pipeline() that returns a scikit-learn pipeline for consistent integration.
"""
from generated_pipelines.pipeline_iris_clean import create_model_pipeline


def build_full_pipeline():
    """Return scikit-learn pipeline built by the generated script."""
    return create_model_pipeline()


if __name__ == '__main__':
    # Quick smoke test
    pl = build_full_pipeline()
    print('Pipeline built:', pl)
