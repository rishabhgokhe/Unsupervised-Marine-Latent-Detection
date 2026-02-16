from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    missing_ratio_before: Dict[str, float]
    missing_ratio_after: Dict[str, float]
    outlier_clip_bounds: Dict[str, tuple[float, float]]


def encode_directional_features(df: pd.DataFrame, directional_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in directional_columns:
        if col not in out.columns:
            continue
        radians = np.deg2rad(out[col])
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    return out


def handle_missing_values(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    numeric_columns: List[str],
    small_gap_limit: int,
    medium_gap_limit: int,
    drop_large_gap_rows: bool,
) -> pd.DataFrame:
    out = df.copy()

    def _fill_group(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values(timestamp_col).copy() if timestamp_col in grp.columns else grp.copy()
        # short gaps: directional carry-forward/backward for continuity
        g[numeric_columns] = g[numeric_columns].ffill(limit=small_gap_limit).bfill(limit=small_gap_limit)
        # medium gaps: linear interpolation with bounded window
        g[numeric_columns] = g[numeric_columns].interpolate(method="linear", limit=medium_gap_limit)
        return g

    out = out.groupby(group_col, group_keys=False).apply(_fill_group)

    if drop_large_gap_rows:
        out = out.dropna(subset=numeric_columns)

    return out.reset_index(drop=True)


def clip_outliers(
    df: pd.DataFrame,
    numeric_columns: List[str],
    low_q: float,
    high_q: float,
) -> tuple[pd.DataFrame, Dict[str, tuple[float, float]]]:
    out = df.copy()
    bounds: Dict[str, tuple[float, float]] = {}

    for col in numeric_columns:
        q_low = out[col].quantile(low_q)
        q_high = out[col].quantile(high_q)
        bounds[col] = (float(q_low), float(q_high))
        out[col] = out[col].clip(lower=q_low, upper=q_high)

    return out, bounds


def quality_preprocess(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    numeric_columns: List[str],
    directional_columns: List[str],
    small_gap_limit: int,
    medium_gap_limit: int,
    drop_large_gap_rows: bool,
    low_q: float,
    high_q: float,
) -> tuple[pd.DataFrame, QualityReport]:
    missing_before = {c: float(df[c].isna().mean()) for c in numeric_columns}
    fill_columns = [c for c in [*numeric_columns, *directional_columns] if c in df.columns]

    working = handle_missing_values(
        df,
        group_col=group_col,
        timestamp_col=timestamp_col,
        numeric_columns=fill_columns,
        small_gap_limit=small_gap_limit,
        medium_gap_limit=medium_gap_limit,
        drop_large_gap_rows=drop_large_gap_rows,
    )
    working = encode_directional_features(working, directional_columns)
    working, clip_bounds = clip_outliers(working, numeric_columns, low_q=low_q, high_q=high_q)

    missing_after = {c: float(working[c].isna().mean()) for c in numeric_columns}
    report = QualityReport(
        missing_ratio_before=missing_before,
        missing_ratio_after=missing_after,
        outlier_clip_bounds=clip_bounds,
    )
    return working, report
