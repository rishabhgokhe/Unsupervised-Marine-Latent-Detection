from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


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
    records = []
    meta_records = []

    for station, grp in df.groupby(station_col, sort=False):
        g = grp.sort_values(timestamp_col).reset_index(drop=True)
        if len(g) < window_size:
            continue

        for start in range(0, len(g) - window_size + 1, step_size):
            end = start + window_size
            w = g.iloc[start:end]
            rec = _extract_stats(w, feature_columns, stats)
            records.append(rec)
            meta_records.append(
                {
                    "station": station,
                    "start_idx": start,
                    "end_idx": end - 1,
                    "start_time": w[timestamp_col].iloc[0],
                    "end_time": w[timestamp_col].iloc[-1],
                }
            )

    if not records:
        raise ValueError("No windows could be generated; check window_size/step_size and data length")

    X = pd.DataFrame.from_records(records)
    meta = pd.DataFrame.from_records(meta_records)
    return WindowedFeatures(X=X, meta=meta)
