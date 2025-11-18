from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mle_star_ablation import mle_star_generated_pipeline as mgp
from src.mle_star_ablation.config import get_standard_configs


def _has_non_one_n_jobs(est):
    try:
        params = est.get_params()
        if 'n_jobs' in params:
            return params['n_jobs'] != 1
    except Exception:
        pass
    # Drill into pipelines and column transformers
    if hasattr(est, 'steps'):
        for _, nested in est.steps:
            if _has_non_one_n_jobs(nested):
                return True
    if hasattr(est, 'transformers'):
        for _, transformer, _ in est.transformers:
            if _has_non_one_n_jobs(transformer):
                return True
    return False


def test_deterministic_sets_n_jobs_to_one_if_present():
    configs = get_standard_configs()
    # pick a config that typically uses GridSearchCV (full)
    cfg = next((c for c in configs if c.name == 'full'), configs[0])
    # Build deterministic pipeline
    p = mgp.build_pipeline(cfg, random_state=42, deterministic=True)
    # Traverse and ensure no estimator has n_jobs != 1
    assert not _has_non_one_n_jobs(p)
