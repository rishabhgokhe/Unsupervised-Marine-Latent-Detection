from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.config import ProjectConfig
from src.data.ingest import resample_by_station
from src.data.preprocess import quality_preprocess, replace_special_missing
from src.features.window_engine import WindowedFeatures, generate_multiscale_window_features
from src.features.windows import build_sliding_windows


@dataclass
class PreprocessOutput:
    processed: pd.DataFrame
    windowed: WindowedFeatures
    x_scaled: np.ndarray


def preprocess_input(df: pd.DataFrame, cfg: ProjectConfig, scaler: object, inference_config: Dict | None = None) -> PreprocessOutput:
    inference_config = inference_config or {}

    numeric_columns: List[str] = list(inference_config.get("numeric_columns", cfg.data.numeric_columns))
    directional_columns: List[str] = list(inference_config.get("directional_columns", cfg.data.directional_columns))

    base_df = df.copy()
    base_df = replace_special_missing(base_df, [*numeric_columns, *directional_columns], sentinels=[99, 999, 9999, 99999])

    aligned = resample_by_station(
        base_df,
        station_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=numeric_columns,
        rule=cfg.data.resample_rule,
        directional_columns=directional_columns,
    )

    processed, _ = quality_preprocess(
        aligned,
        group_col=cfg.data.station_col,
        timestamp_col=cfg.data.timestamp_col,
        numeric_columns=numeric_columns,
        directional_columns=directional_columns,
        small_gap_limit=cfg.preprocess.small_gap_limit,
        medium_gap_limit=cfg.preprocess.medium_gap_limit,
        drop_large_gap_rows=cfg.preprocess.drop_large_gap_rows,
        low_q=cfg.preprocess.clip_quantile_low,
        high_q=cfg.preprocess.clip_quantile_high,
    )

    model_feature_cols = list(inference_config.get("model_feature_cols", []))
    if not model_feature_cols:
        directional_encoded = [f"{c}_SIN" for c in directional_columns] + [f"{c}_COS" for c in directional_columns]
        model_feature_cols = list(dict.fromkeys([*numeric_columns, *[c for c in directional_encoded if c in processed.columns]]))

    window_cfg = inference_config.get("windowing", {})
    use_multiscale = bool(window_cfg.get("use_multiscale", cfg.features.use_multiscale))

    if use_multiscale:
        windowed = generate_multiscale_window_features(
            processed,
            station_col=cfg.data.station_col,
            timestamp_col=cfg.data.timestamp_col,
            feature_columns=model_feature_cols,
            window_sizes=list(window_cfg.get("multi_window_sizes", cfg.features.multi_window_sizes)),
            stride=int(window_cfg.get("multi_stride", cfg.features.multi_stride)),
        )
    else:
        windowed = build_sliding_windows(
            processed,
            station_col=cfg.data.station_col,
            timestamp_col=cfg.data.timestamp_col,
            feature_columns=model_feature_cols,
            window_size=int(window_cfg.get("window_size", cfg.features.window_size)),
            step_size=int(window_cfg.get("step_size", cfg.features.step_size)),
            stats=list(window_cfg.get("rolling_features", cfg.features.rolling_features)),
        )

    x_scaled = scaler.transform(windowed.X.values)
    return PreprocessOutput(processed=processed, windowed=windowed, x_scaled=x_scaled)
