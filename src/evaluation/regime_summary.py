from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def infer_semantic_regime_names(window_df: pd.DataFrame) -> Dict[int, str]:
    if window_df.empty:
        return {}

    score_cols = [c for c in ["WAVE_HGT_mean", "WIND_SPEED_mean", "SEA_LVL_PRES_mean"] if c in window_df.columns]
    if not score_cols:
        score_cols = [c for c in window_df.columns if c.endswith("_mean")][:5]

    regime_scores = (
        window_df.groupby("regime_id")[score_cols]
        .mean()
        .assign(severity=lambda d: d.mean(axis=1))
        .sort_values("severity")
    )

    ordered_ids = list(regime_scores.index)
    names = {}
    palette = ["calm regime", "rough sea regime", "storm regime"]
    for idx, rid in enumerate(ordered_ids):
        if idx < len(palette):
            names[int(rid)] = palette[idx]
        else:
            names[int(rid)] = f"high-variance regime {idx + 1}"
    return names


def build_window_regime_frame(meta: pd.DataFrame, window_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    out["regime_id"] = labels.astype(int)

    feature_cols = [c for c in window_features.columns if c.endswith("_mean")]
    for col in feature_cols:
        out[col] = window_features[col].values

    out["duration_hours"] = (pd.to_datetime(out["end_time"]) - pd.to_datetime(out["start_time"])) / pd.Timedelta(hours=1)
    return out


def regime_summary_table(window_df: pd.DataFrame, selected_feature_cols: List[str]) -> pd.DataFrame:
    base_cols = [c for c in selected_feature_cols if c in window_df.columns]
    agg = window_df.groupby(["regime_id", "regime_name"], as_index=False).agg(
        n_windows=("regime_id", "size"),
        avg_window_duration_hr=("duration_hours", "mean"),
        **{f"mean_{c}": (c, "mean") for c in base_cols},
    )
    return agg.sort_values("n_windows", ascending=False).reset_index(drop=True)
