from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class WindowedFeatures:
    X: pd.DataFrame
    meta: pd.DataFrame


def compute_window_statistics(window: np.ndarray) -> np.ndarray:
    """Return mean/std/trend/energy/max/min per feature for one window."""
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    trend = window[-1] - window[0]
    energy = np.sum(window**2, axis=0) / max(len(window), 1)
    max_val = window.max(axis=0)
    min_val = window.min(axis=0)
    return np.concatenate([mean, std, trend, energy, max_val, min_val])


def _feature_names(columns: Iterable[str], scale: int) -> List[str]:
    stats = ["mean", "std", "trend", "energy", "max", "min"]
    names: List[str] = []
    for stat in stats:
        for col in columns:
            names.append(f"{col}_s{scale}_{stat}")
    return names


def generate_multiscale_window_features(
    df: pd.DataFrame,
    station_col: str,
    timestamp_col: str,
    feature_columns: List[str],
    window_sizes: List[int],
    stride: int,
) -> WindowedFeatures:
    """Generate past-only windows at multiple scales and stack them."""
    records: list[dict] = []
    meta_records: list[dict] = []

    for station, grp in df.groupby(station_col, sort=False):
        g = grp.sort_values(timestamp_col).reset_index(drop=True)
        values = g[feature_columns].to_numpy(dtype=float)

        for scale in sorted(window_sizes):
            if len(g) < scale:
                continue

            cols = _feature_names(feature_columns, scale)
            for start in range(0, len(g) - scale + 1, stride):
                end = start + scale
                window = values[start:end]
                stats = compute_window_statistics(window)

                rec = {cols[i]: float(stats[i]) for i in range(len(cols))}
                records.append(rec)
                meta_records.append(
                    {
                        "station": station,
                        "window_scale": int(scale),
                        "start_idx": int(start),
                        "end_idx": int(end - 1),
                        "start_time": g[timestamp_col].iloc[start],
                        "end_time": g[timestamp_col].iloc[end - 1],
                    }
                )

    if not records:
        raise ValueError("No windows generated; verify window_sizes/stride and data length")

    X = pd.DataFrame.from_records(records).fillna(0.0)
    meta = pd.DataFrame.from_records(meta_records)
    return WindowedFeatures(X=X, meta=meta)
