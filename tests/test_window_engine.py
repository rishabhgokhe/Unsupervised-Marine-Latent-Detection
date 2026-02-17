import pandas as pd

from src.features.window_engine import generate_multiscale_window_features


def test_generate_multiscale_window_features():
    df = pd.DataFrame(
        {
            "STATION": ["A"] * 10,
            "DATE": pd.date_range("2023-01-01", periods=10, freq="1H"),
            "WIND_SPEED": list(range(10)),
            "WAVE_HGT": [0.5 + 0.1 * i for i in range(10)],
        }
    )

    out = generate_multiscale_window_features(
        df,
        station_col="STATION",
        timestamp_col="DATE",
        feature_columns=["WIND_SPEED", "WAVE_HGT"],
        window_sizes=[3, 6],
        stride=2,
    )

    assert len(out.X) > 0
    assert "WIND_SPEED_s3_mean" in out.X.columns
    assert "WAVE_HGT_s6_energy" in out.X.columns
    assert "window_scale" in out.meta.columns
