import pandas as pd

from src.features.windows import build_sliding_windows


def test_build_sliding_windows_basic():
    df = pd.DataFrame(
        {
            "STATION": ["A"] * 6,
            "DATE": pd.date_range("2023-01-01", periods=6, freq="1H"),
            "WIND_SPEED": [1, 2, 3, 4, 5, 6],
            "SEA_LVL_PRES": [1000, 1001, 1002, 1003, 1004, 1005],
        }
    )

    out = build_sliding_windows(
        df,
        station_col="STATION",
        timestamp_col="DATE",
        feature_columns=["WIND_SPEED", "SEA_LVL_PRES"],
        window_size=3,
        step_size=2,
        stats=["mean", "std", "min", "max"],
    )

    assert len(out.X) == 2
    assert len(out.meta) == 2
    assert "WIND_SPEED_s3_mean" in out.X.columns
    assert "SEA_LVL_PRES_s3_max" in out.X.columns
