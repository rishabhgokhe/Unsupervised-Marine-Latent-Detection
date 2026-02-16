from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    input_path: str
    timestamp_col: str = "DATE"
    station_col: str = "STATION"
    numeric_columns: List[str] = field(default_factory=list)
    directional_columns: List[str] = field(default_factory=list)
    resample_rule: str = "1H"


@dataclass
class PreprocessConfig:
    small_gap_limit: int = 2
    medium_gap_limit: int = 6
    drop_large_gap_rows: bool = False
    clip_quantile_low: float = 0.005
    clip_quantile_high: float = 0.995


@dataclass
class FeatureConfig:
    window_size: int = 12
    step_size: int = 6
    rolling_features: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])


@dataclass
class ModelConfig:
    candidate_states: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    random_state: int = 42
    hmm_covariance_type: str = "diag"
    min_segment_length: int = 3
    n_super_regimes: int = 3


@dataclass
class EvalConfig:
    boundary_tolerance_steps: int = 2


@dataclass
class AppConfig:
    app_title: str = "Marine Regime Discovery"
    author: str = "Rishabh Gokhe"


@dataclass
class TrackingConfig:
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "marine-regime-discovery"
    run_name: str = "baseline"
    log_artifacts: bool = True


@dataclass
class DeepConfig:
    enabled: bool = False
    enable_vae: bool = False
    seq_len: int = 8
    hidden_dim: int = 64
    latent_dim: int = 32
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    vae_latent_dim: int = 8
    vae_beta: float = 1.0


@dataclass
class ProjectConfig:
    data: DataConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    app: AppConfig = field(default_factory=AppConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)


def _get(section: Dict[str, Any], key: str, default: Any) -> Any:
    return section.get(key, default) if section else default


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    data_raw = raw.get("data", {})
    preprocess_raw = raw.get("preprocess", {})
    feature_raw = raw.get("features", {})
    model_raw = raw.get("models", {})
    eval_raw = raw.get("evaluation", {})
    app_raw = raw.get("app", {})
    tracking_raw = raw.get("tracking", {})
    deep_raw = raw.get("deep", {})

    data_cfg = DataConfig(
        input_path=_get(data_raw, "input_path", ""),
        timestamp_col=_get(data_raw, "timestamp_col", "DATE"),
        station_col=_get(data_raw, "station_col", "STATION"),
        numeric_columns=list(_get(data_raw, "numeric_columns", [])),
        directional_columns=list(_get(data_raw, "directional_columns", [])),
        resample_rule=_get(data_raw, "resample_rule", "1H"),
    )

    if not data_cfg.input_path:
        raise ValueError("config.data.input_path is required")

    return ProjectConfig(
        data=data_cfg,
        preprocess=PreprocessConfig(
            small_gap_limit=int(_get(preprocess_raw, "small_gap_limit", 2)),
            medium_gap_limit=int(_get(preprocess_raw, "medium_gap_limit", 6)),
            drop_large_gap_rows=bool(_get(preprocess_raw, "drop_large_gap_rows", False)),
            clip_quantile_low=float(_get(preprocess_raw, "clip_quantile_low", 0.005)),
            clip_quantile_high=float(_get(preprocess_raw, "clip_quantile_high", 0.995)),
        ),
        features=FeatureConfig(
            window_size=int(_get(feature_raw, "window_size", 12)),
            step_size=int(_get(feature_raw, "step_size", 6)),
            rolling_features=list(_get(feature_raw, "rolling_features", ["mean", "std", "min", "max"])),
        ),
        models=ModelConfig(
            candidate_states=list(_get(model_raw, "candidate_states", [2, 3, 4, 5])),
            random_state=int(_get(model_raw, "random_state", 42)),
            hmm_covariance_type=str(_get(model_raw, "hmm_covariance_type", "diag")),
            min_segment_length=int(_get(model_raw, "min_segment_length", 3)),
            n_super_regimes=int(_get(model_raw, "n_super_regimes", 3)),
        ),
        evaluation=EvalConfig(
            boundary_tolerance_steps=int(_get(eval_raw, "boundary_tolerance_steps", 2)),
        ),
        app=AppConfig(
            app_title=str(_get(app_raw, "app_title", "Marine Regime Discovery")),
            author=str(_get(app_raw, "author", "Rishabh Gokhe")),
        ),
        tracking=TrackingConfig(
            enabled=bool(_get(tracking_raw, "enabled", False)),
            tracking_uri=str(_get(tracking_raw, "tracking_uri", "file:./mlruns")),
            experiment_name=str(_get(tracking_raw, "experiment_name", "marine-regime-discovery")),
            run_name=str(_get(tracking_raw, "run_name", "baseline")),
            log_artifacts=bool(_get(tracking_raw, "log_artifacts", True)),
        ),
        deep=DeepConfig(
            enabled=bool(_get(deep_raw, "enabled", False)),
            enable_vae=bool(_get(deep_raw, "enable_vae", False)),
            seq_len=int(_get(deep_raw, "seq_len", 8)),
            hidden_dim=int(_get(deep_raw, "hidden_dim", 64)),
            latent_dim=int(_get(deep_raw, "latent_dim", 32)),
            epochs=int(_get(deep_raw, "epochs", 20)),
            batch_size=int(_get(deep_raw, "batch_size", 64)),
            learning_rate=float(_get(deep_raw, "learning_rate", 0.001)),
            vae_latent_dim=int(_get(deep_raw, "vae_latent_dim", 8)),
            vae_beta=float(_get(deep_raw, "vae_beta", 1.0)),
        ),
    )
