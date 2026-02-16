from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(
        self,
        enabled: bool,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        log_artifacts: bool,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_artifacts = log_artifacts
        self._mlflow = None
        self._run = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import mlflow
        except Exception:
            logger.warning("MLflow not installed; tracking disabled")
            self.enabled = False
            return

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        for key, value in params.items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact_dir(self, artifact_dir: str | Path) -> None:
        if not self.enabled or self._mlflow is None or not self.log_artifacts:
            return
        self._mlflow.log_artifacts(str(artifact_dir))

    def end(self) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.end_run()
