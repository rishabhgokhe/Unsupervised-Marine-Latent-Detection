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


def replace_special_missing(df: pd.DataFrame, columns: List[str], sentinels: List[float]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    if not cols:
        return out
    out[cols] = out[cols].replace(sentinels, np.nan)
    return out


def encode_directional_features(df: pd.DataFrame, directional_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in directional_columns:
        if col not in out.columns:
            continue
        radians = np.deg2rad(out[col])
        out[f"{col}_SIN"] = np.sin(radians)
        out[f"{col}_COS"] = np.cos(radians)
    return out


def handle_missing_values(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    numeric_columns: List[str],
    directional_columns: List[str] | None,
    small_gap_limit: int,
    medium_gap_limit: int,
    drop_large_gap_rows: bool,
) -> pd.DataFrame:
    out = df.copy()
    directional_columns = directional_columns or []

    def _fill_group(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values(timestamp_col).copy() if timestamp_col in grp.columns else grp.copy()
        present_directional = [c for c in directional_columns if c in g.columns]
        present_numeric = [c for c in numeric_columns if c in g.columns and c not in present_directional]
        # short gaps: directional carry-forward/backward for continuity
        if present_numeric:
            g[present_numeric] = g[present_numeric].ffill(limit=small_gap_limit).bfill(limit=small_gap_limit)
            # medium gaps: linear interpolation with bounded window
            g[present_numeric] = g[present_numeric].interpolate(method="linear", limit=medium_gap_limit)

        # Directional interpolation via sin/cos to preserve circular continuity.
        for dcol in present_directional:
            rad = np.deg2rad(g[dcol])
            sin = np.sin(rad)
            cos = np.cos(rad)
            sin = sin.ffill(limit=small_gap_limit).bfill(limit=small_gap_limit)
            cos = cos.ffill(limit=small_gap_limit).bfill(limit=small_gap_limit)
            sin = sin.interpolate(method="linear", limit=medium_gap_limit)
            cos = cos.interpolate(method="linear", limit=medium_gap_limit)
            g[dcol] = (np.rad2deg(np.arctan2(sin, cos)) + 360) % 360
        return g

    grouped = out.groupby(group_col, group_keys=False)
    try:
        # Pandas >= 2.2 supports include_groups to silence deprecation warning.
        out = grouped.apply(_fill_group, include_groups=False)
    except TypeError:
        out = grouped.apply(_fill_group)

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
        directional_columns=directional_columns,
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
