"""
Compute an aggregated best-vs-full statistical summary for the N-run ablation suite.

The script reads the *actual* artifact files produced by `scripts/run_ablation.py` / `scripts/run_experiment_suite.py`:
- `results/*/*/summary_statistics_*.csv` (means/std/CI, sorted by mean)
- `results/*/*/pairwise_comparisons_*.csv` (paired t-test, Bonferroni, Cohen's d)

Output:
- `results/statistical_analysis.csv` (one row per dataset/experiment)

This file is referenced by the report as a compact replication artifact. Do not hardcode numbers here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    pipeline_wrapper: str
    task_type: str
    n_runs: int


def cohens_d(mean1, std1, n1, mean2, std2, n2):
    """Calculate Cohen's d effect size."""
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def cohens_d_ci(d, n1, n2, alpha=0.05):
    """Calculate 95% CI for Cohen's d using non-central t approximation."""
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    t_crit = stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)
    return (d - t_crit * se_d, d + t_crit * se_d)


def _parse_timestamp(path: Path, prefix: str) -> str | None:
    m = re.search(rf"{re.escape(prefix)}_(\d{{8}}_\d{{6}})\.csv$", path.name)
    return m.group(1) if m else None


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_p_value(p):
    """Interpret p-value significance."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def _load_suite_specs(repo_root: Path) -> list[ExperimentSpec]:
    config_path = repo_root / "configs" / "experiment_suite.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    global_n_runs = int(config.get("global", {}).get("n_runs", 20))

    specs: list[ExperimentSpec] = []
    for exp in config.get("experiments", []):
        if not exp.get("enabled", False):
            continue
        name = str(exp["name"])
        pipeline_file = Path(str(exp["pipeline_file"]))
        wrapper = pipeline_file.stem
        task_type = str(exp.get("task_type", "unknown"))
        n_runs = int(exp.get("n_runs", global_n_runs))
        specs.append(ExperimentSpec(name=name, pipeline_wrapper=wrapper, task_type=task_type, n_runs=n_runs))

    return specs


def _pick_latest_summary(results_dir: Path, expected_n_runs: int) -> tuple[Path, str]:
    candidates = sorted(results_dir.glob("summary_statistics_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No summary_statistics_*.csv found in {results_dir}")

    usable: list[tuple[str, Path]] = []
    fallback: list[tuple[str, Path]] = []

    for path in candidates:
        ts = _parse_timestamp(path, "summary_statistics") or "00000000_000000"
        fallback.append((ts, path))
        try:
            df = pd.read_csv(path)
            full = df[df["configuration"] == "full"]
            if not full.empty and int(full.iloc[0]["n_runs"]) == expected_n_runs:
                usable.append((ts, path))
        except Exception:
            continue

    picked = max(usable or fallback, key=lambda x: x[0])
    return picked[1], picked[0]


def _find_pairwise_for_timestamp(results_dir: Path, ts: str) -> Path:
    direct = results_dir / f"pairwise_comparisons_{ts}.csv"
    if direct.exists():
        return direct
    candidates = sorted(results_dir.glob("pairwise_comparisons_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No pairwise_comparisons_*.csv found in {results_dir}")
    return max(candidates, key=lambda p: _parse_timestamp(p, "pairwise_comparisons") or p.name)


def _analyze_experiment(spec: ExperimentSpec, repo_root: Path) -> dict:
    results_dir = repo_root / "results" / spec.name / spec.pipeline_wrapper

    summary_path, ts = _pick_latest_summary(results_dir, expected_n_runs=spec.n_runs)
    pairwise_path = _find_pairwise_for_timestamp(results_dir, ts)

    summary_df = pd.read_csv(summary_path)
    full_row = summary_df[summary_df["configuration"] == "full"].iloc[0]
    best_row = (
        summary_df[~summary_df["configuration"].isin(["full", "baseline"])]
        .sort_values("mean", ascending=False)
        .iloc[0]
    )

    best_cfg = str(best_row["configuration"])
    full_mean = float(full_row["mean"])
    best_mean = float(best_row["mean"])

    pairwise_df = pd.read_csv(pairwise_path)
    mask = (
        ((pairwise_df["config_a"] == best_cfg) & (pairwise_df["config_b"] == "full"))
        | ((pairwise_df["config_a"] == "full") & (pairwise_df["config_b"] == best_cfg))
    )
    if not mask.any():
        raise ValueError(f"Missing best-vs-full row in {pairwise_path} for best='{best_cfg}'")
    comp = pairwise_df[mask].iloc[0]

    t_stat = pd.to_numeric(comp.get("t_statistic"), errors="coerce")
    p_value = pd.to_numeric(comp.get("p_value"), errors="coerce")
    d = pd.to_numeric(comp.get("cohen_d"), errors="coerce")

    if str(comp.get("config_a")) == "full":
        # Convert to best-full direction for consistency with report tables.
        t_stat = -t_stat
        d = -d

    n = int(full_row["n_runs"])
    d_ci_lower, d_ci_upper = (np.nan, np.nan)
    if not np.isnan(d):
        d_ci_lower, d_ci_upper = cohens_d_ci(float(d), n, n)

    delta = best_mean - full_mean

    return {
        "dataset": spec.name,
        "task_type": spec.task_type,
        "best_config": best_cfg,
        "best_score": best_mean,
        "best_std": float(best_row["std"]),
        "full_score": full_mean,
        "full_std": float(full_row["std"]),
        "delta": delta,
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "p_interpret": interpret_p_value(float(p_value)) if not np.isnan(p_value) else "—",
        "cohens_d": float(d) if not np.isnan(d) else np.nan,
        "d_ci_lower": float(d_ci_lower) if not np.isnan(d_ci_lower) else np.nan,
        "d_ci_upper": float(d_ci_upper) if not np.isnan(d_ci_upper) else np.nan,
        "d_interpret": interpret_cohens_d(float(d)) if not np.isnan(d) else "",
        "n": n,
        "bonferroni_significant": bool(comp.get("bonferroni_significant")) if "bonferroni_significant" in comp else "",
        "corrected_alpha": float(comp.get("corrected_alpha")) if pd.notna(comp.get("corrected_alpha")) else np.nan,
        "source_summary": str(summary_path.relative_to(repo_root)),
        "source_pairwise": str(pairwise_path.relative_to(repo_root)),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]

    specs = _load_suite_specs(repo_root)
    if not specs:
        raise RuntimeError("No enabled experiments found in configs/experiment_suite.yaml")

    results_list = [_analyze_experiment(spec, repo_root) for spec in specs]
    df = pd.DataFrame(results_list)

    output_path = repo_root / "results" / "statistical_analysis.csv"
    df.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Datasets analyzed: {len(df)}")
    print(df[['dataset','best_config','delta','p_interpret','bonferroni_significant']].to_string(index=False))
    
    # Average Cohen's d
    print(f"  Mean Cohen's d: {df['cohens_d'].mean():.2f}")
    print(f"  Median Cohen's d: {df['cohens_d'].median():.2f}")
    print(f"  Mean Δ (score improvement): {df['delta'].mean():.3f}")
    print()
    
    # Generate markdown table for report
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE FOR REPORT")
    print("=" * 80)
    print()
    print("**Таблиця 2. Статистичний аналіз переваги спрощених конфігурацій (N=20)**")
    print()
    print("| Датасет | Найкраща | Score±Std | Full±Std | Δ | t | p-value | d | 95% CI | Ефект |")
    print("|---------|----------|-----------|----------|---|---|---------|---|--------|-------|")
    
    for _, row in df.iterrows():
        dataset_short = row['dataset'].replace('cls_', '').replace('reg_', '')
        print(f"| {dataset_short} | `{row['best_config']}` | "
              f"{row['best_score']:.3f}±{row['best_std']:.3f} | "
              f"{row['full_score']:.3f}±{row['full_std']:.3f} | "
              f"{row['delta']:+.3f} | "
              f"{row['t_statistic']:.2f} | "
              f"{row['p_value']:.2e}{row['p_interpret']} | "
              f"{row['cohens_d']:.2f} | "
              f"[{row['d_ci_lower']:.2f}, {row['d_ci_upper']:.2f}] | "
              f"**{row['d_interpret']}** |")
    
    print()
    print("*Значущість: *** p<0.001, ** p<0.01, * p<0.05, ns - незначущий*")
    print("*Класифікація Cohen's d: Large (d≥0.8), Medium (0.5≤d<0.8), Small (0.2≤d<0.5), Negligible (d<0.2)*")
    print()
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / 'results' / 'statistical_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    main()
