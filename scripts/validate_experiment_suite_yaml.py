"""Validate that configs/experiment_suite.yaml parses correctly.

Usage:
  python scripts/validate_experiment_suite_yaml.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: PyYAML not available: {exc}")
        return 2

    root = Path(__file__).resolve().parents[1]
    manifest = root / "configs" / "experiment_suite.yaml"
    if not manifest.exists():
        print(f"ERROR: Not found: {manifest}")
        return 2

    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "experiments" not in data:
        print("ERROR: YAML parsed but does not look like an experiment manifest")
        return 1

    experiments = data.get("experiments") or []
    enabled = [e for e in experiments if isinstance(e, dict) and e.get("enabled")]
    print(f"OK: {manifest}")
    print(f"Experiments: {len(experiments)} | Enabled: {len(enabled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
