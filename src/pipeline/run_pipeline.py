from __future__ import annotations

import argparse
import logging

from src.core.config import load_config
from src.core.logging_utils import configure_logging
from src.core.tracking import ExperimentTracker
from src.pipeline.regime_pipeline import run_pipeline, save_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run marine regime discovery pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="outputs/latest", help="Output artifacts directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)

    cfg = load_config(args.config)
    tracker = ExperimentTracker(
        enabled=cfg.tracking.enabled,
        tracking_uri=cfg.tracking.tracking_uri,
        experiment_name=cfg.tracking.experiment_name,
        run_name=cfg.tracking.run_name,
        log_artifacts=cfg.tracking.log_artifacts,
    )

    tracker.start()
    try:
        tracker.log_params(
            {
                "resample_rule": cfg.data.resample_rule,
                "window_size": cfg.features.window_size,
                "step_size": cfg.features.step_size,
                "candidate_states": ",".join(map(str, cfg.models.candidate_states)),
                "n_super_regimes": cfg.models.n_super_regimes,
                "deep_enabled": cfg.deep.enabled,
                "deep_enable_vae": cfg.deep.enable_vae,
            }
        )

        result = run_pipeline(cfg)
        save_artifacts(result, args.output)

        for model_name, model_metrics in result.model_metrics.items():
            flat = {f"{model_name}.{k}": float(v) for k, v in model_metrics.items()}
            tracker.log_metrics(flat)
        tracker.log_artifact_dir(args.output)
    finally:
        tracker.end()


if __name__ == "__main__":
    main()
