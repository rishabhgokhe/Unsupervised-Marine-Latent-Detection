import numpy as np
import pandas as pd

from src.data.preprocess import quality_preprocess


def test_quality_preprocess_encodes_direction_and_fills_missing():
    df = pd.DataFrame(
        {
            "STATION": ["A", "A", "A", "A"],
            "DATE": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 01:00:00",
                    "2023-01-01 02:00:00",
                    "2023-01-01 03:00:00",
                ]
            ),
            "WIND_SPEED": [1.0, np.nan, 3.0, 4.0],
            "WIND_DIR": [0.0, 90.0, 180.0, 270.0],
        }
    )

    out, report = quality_preprocess(
        df,
        group_col="STATION",
        timestamp_col="DATE",
        numeric_columns=["WIND_SPEED"],
        directional_columns=["WIND_DIR"],
        small_gap_limit=2,
        medium_gap_limit=2,
        drop_large_gap_rows=False,
        low_q=0.0,
        high_q=1.0,
    )

    assert "WIND_DIR_sin" in out.columns
    assert "WIND_DIR_cos" in out.columns
    assert float(out["WIND_SPEED"].isna().mean()) == 0.0
    assert report.missing_ratio_before["WIND_SPEED"] > report.missing_ratio_after["WIND_SPEED"]
