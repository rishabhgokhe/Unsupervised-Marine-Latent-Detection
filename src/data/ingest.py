from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


class DataIngestionError(RuntimeError):
    pass


def read_dataset_file(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise DataIngestionError(f"Input dataset not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    raise DataIngestionError(f"Unsupported dataset format: {suffix}. Expected .csv or .parquet")


def read_csv_dataset(
    path: str | Path,
    timestamp_col: str,
    station_col: str,
    expected_columns: Iterable[str],
    strict_columns: bool = False,
) -> pd.DataFrame:
    df = read_dataset_file(path)
    required = {timestamp_col, station_col, *expected_columns}
    missing = sorted(required - set(df.columns))
    if missing and strict_columns:
        raise DataIngestionError(f"Missing required columns: {missing}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        raise DataIngestionError("Dataset has no valid timestamp rows after parsing")

    return df.sort_values([station_col, timestamp_col]).reset_index(drop=True)


def resample_by_station(
    df: pd.DataFrame,
    station_col: str,
    timestamp_col: str,
    numeric_columns: List[str],
    rule: str,
    directional_columns: List[str] | None = None,
) -> pd.DataFrame:
    if not numeric_columns:
        raise ValueError("numeric_columns must be provided for resampling")

    # Pandas deprecates uppercase hour alias; normalize only the hour token.
    if "H" in rule:
        rule = rule.replace("H", "h")

    directional_columns = directional_columns or []
    frames = []
    for station_id, station_df in df.groupby(station_col, sort=False):
        present_numeric = [c for c in numeric_columns if c in station_df.columns]
        present_directional = [c for c in directional_columns if c in station_df.columns]

        subset = station_df[[timestamp_col, *present_numeric, *present_directional]].copy()
        subset = subset.set_index(timestamp_col).sort_index()

        out = pd.DataFrame(index=subset.resample(rule).mean().index)
        if present_numeric:
            out[present_numeric] = subset[present_numeric].resample(rule).mean()

        # Circular mean for directional variables keeps 0/360 continuity intact.
        for dcol in present_directional:
            rad = np.deg2rad(subset[dcol])
            sin_avg = np.sin(rad).resample(rule).mean()
            cos_avg = np.cos(rad).resample(rule).mean()
            out[dcol] = (np.rad2deg(np.arctan2(sin_avg, cos_avg)) + 360) % 360

        out = out.reset_index()
        out[station_col] = station_id
        frames.append(out)

    out_df = pd.concat(frames, ignore_index=True)
    ordered_cols = [station_col, timestamp_col, *[c for c in numeric_columns if c in out_df.columns], *[c for c in directional_columns if c in out_df.columns]]
    return out_df[ordered_cols]
