import os
from pathlib import Path
import sys
import pytest

# Ensure repo root is in sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_generated_pipeline import validate_pipeline_file


def test_validate_known_wrapper():
    # Use an existing wrapper that has been generated during previous runs
    candidate = Path('model_comparison_results/gemini_2.5_flash_lite_iris.py')
    if not candidate.exists():
        pytest.skip('No generated wrapper for iris present to validate')
    ok = validate_pipeline_file(candidate, cv=2, random_state=42)
    assert ok
