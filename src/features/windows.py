from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.features.window_engine import WindowedFeatures as EngineWindowedFeatures
from src.features.window_engine import generate_multiscale_window_features

@dataclass
class WindowedFeatures:
    X: pd.DataFrame
    meta: pd.DataFrame


def _extract_stats(window: pd.DataFrame, columns: List[str], stats: List[str]) -> dict[str, float]:
    feat: dict[str, float] = {}
    for col in columns:
        series = window[col]
        for st in stats:
            key = f"{col}_{st}"
            if st == "mean":
                feat[key] = float(series.mean())
            elif st == "std":
                feat[key] = float(series.std(ddof=0))
            elif st == "min":
                feat[key] = float(series.min())
            elif st == "max":
                feat[key] = float(series.max())
            elif st == "median":
                feat[key] = float(series.median())
            elif st == "energy":
                feat[key] = float(np.sum(series.values ** 2))
            else:
                raise ValueError(f"Unsupported stat: {st}")
    return feat


def build_sliding_windows(
    df: pd.DataFrame,
    station_col: str,
    timestamp_col: str,
    feature_columns: List[str],
    window_size: int,
    step_size: int,
    stats: List[str],
) -> WindowedFeatures:
    # Backward-compatible single-scale API that reuses the multi-scale engine.
    # This keeps notebooks/tests stable while the pipeline uses dedicated multi-scale calls.
    out: EngineWindowedFeatures = generate_multiscale_window_features(
        df=df,
        station_col=station_col,
        timestamp_col=timestamp_col,
        feature_columns=feature_columns,
        window_sizes=[window_size],
        stride=step_size,
    )

    # If caller requested a custom stat subset, filter columns accordingly.
    if stats:
        keep = []
        for c in out.X.columns:
            for st in stats:
                if c.endswith(f"_{st}"):
                    keep.append(c)
                    break
        if keep:
            out = EngineWindowedFeatures(X=out.X[keep].copy(), meta=out.meta)

    return WindowedFeatures(X=out.X, meta=out.meta)
