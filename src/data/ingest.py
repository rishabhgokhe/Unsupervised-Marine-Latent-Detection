from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


class DataIngestionError(RuntimeError):
    pass


def read_csv_dataset(
    path: str | Path,
    timestamp_col: str,
    station_col: str,
    expected_columns: Iterable[str],
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataIngestionError(f"Input dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {timestamp_col, station_col, *expected_columns}
    missing = sorted(required - set(df.columns))
    if missing:
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
) -> pd.DataFrame:
    if not numeric_columns:
        raise ValueError("numeric_columns must be provided for resampling")

    frames = []
    for station_id, station_df in df.groupby(station_col, sort=False):
        subset = station_df[[timestamp_col, *numeric_columns]].copy()
        subset = subset.set_index(timestamp_col).sort_index()
        out = subset.resample(rule).mean().reset_index()
        out[station_col] = station_id
        frames.append(out)

    out_df = pd.concat(frames, ignore_index=True)
    return out_df[[station_col, timestamp_col, *numeric_columns]]
