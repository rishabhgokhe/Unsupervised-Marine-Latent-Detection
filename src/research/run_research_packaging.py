from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import shutil

from src.core.config import load_config
from src.pipeline.regime_pipeline import run_pipeline, save_artifacts


def _best_model(metrics: Dict[str, Dict[str, float]]) -> str:
    if not metrics:
        return "unknown"
    best_name = None
    best_score = None
    for name, vals in metrics.items():
        score = vals.get("silhouette_post")
        if score is None:
            score = vals.get("silhouette")
        if score is None:
            score = vals.get("silhouette_embed")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
    return best_name or next(iter(metrics.keys()))


def _metric_value(metrics: Dict[str, float], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _build_comparative_rows(
    experiment_id: str,
    model_metrics: Dict[str, Dict[str, float]],
    diagnostics: Dict[str, Any] | None,
) -> pd.DataFrame:
    transition_entropy = {}
    if diagnostics:
        transition_entropy = diagnostics.get("transition_entropy", {}) or {}

    all_models = set(model_metrics.keys()) | set(transition_entropy.keys())
    rows: List[Dict[str, Any]] = []
    for model in sorted(all_models):
        metrics = model_metrics.get(model, {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "silhouette": _metric_value(metrics, ["silhouette_post", "silhouette", "silhouette_embed"]),
                "bic": _metric_value(metrics, ["bic"]),
                "avg_duration": _metric_value(metrics, ["mean_regime_duration"]),
                "transition_entropy": transition_entropy.get(model),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(config_path: str, experiment_id: str, output_root: str, notes: str | None) -> None:
    cfg = load_config(config_path)
    result = run_pipeline(cfg)

    exp_dir = Path(output_root) / experiment_id
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts(result, artifacts_dir)
    shutil.copy2(config_path, artifacts_dir / "config.yaml")

    experiment = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes or "",
        "tags": {},
        "selected_model": _best_model(result.model_metrics),
        "config": asdict(cfg),
        "model_metrics": result.model_metrics,
        "quality_report": result.quality_report,
        "diagnostics": result.diagnostics or {},
    }

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    comparative = _build_comparative_rows(experiment_id, result.model_metrics, result.diagnostics)
    comparative.to_csv(exp_dir / "comparative_rows.csv", index=False)


def build_comparative(experiments_dir: str) -> None:
    root = Path(experiments_dir)
    rows = []
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        comp = exp_dir / "comparative_rows.csv"
        if comp.exists():
            rows.append(pd.read_csv(comp))
    if not rows:
        raise FileNotFoundError("No comparative_rows.csv files found under experiments directory.")

    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(root / "comparative_results.csv", index=False)

    mean_by_model = (
        all_rows.groupby("model", as_index=False)
        .agg(
            silhouette=("silhouette", "mean"),
            bic=("bic", "mean"),
            avg_duration=("avg_duration", "mean"),
            transition_entropy=("transition_entropy", "mean"),
        )
    )
    mean_by_model.to_csv(root / "comparative_results_mean_by_model.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research packaging utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    run_exp = sub.add_parser("run-experiment", help="Run one experiment and log outputs")
    run_exp.add_argument("--config", required=True, help="Path to YAML config")
    run_exp.add_argument("--experiment-id", required=True, help="Experiment id")
    run_exp.add_argument("--output-root", default="experiments", help="Root directory for experiments")
    run_exp.add_argument("--notes", default="", help="Notes for experiment.json")

    build_comp = sub.add_parser("build-comparative", help="Build comparative CSVs across experiments")
    build_comp.add_argument("--experiments-dir", default="experiments", help="Experiments directory")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-experiment":
        run_experiment(args.config, args.experiment_id, args.output_root, args.notes)
    elif args.command == "build-comparative":
        build_comparative(args.experiments_dir)


if __name__ == "__main__":
    main()
