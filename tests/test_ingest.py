import pandas as pd

from src.data.ingest import read_csv_dataset, resample_by_station


def test_read_csv_dataset_and_parse_timestamp(tmp_path):
    path = tmp_path / "marine.csv"
    df = pd.DataFrame(
        {
            "STATION": ["A", "A", "A"],
            "DATE": ["2023-01-01 00:00:00", "bad", "2023-01-01 02:00:00"],
            "WIND_SPEED": [1.0, 2.0, 3.0],
            "SEA_LVL_PRES": [1000.0, 1002.0, 1004.0],
        }
    )
    df.to_csv(path, index=False)

    out = read_csv_dataset(
        path=path,
        timestamp_col="DATE",
        station_col="STATION",
        expected_columns=["WIND_SPEED", "SEA_LVL_PRES"],
    )

    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_resample_by_station_hourly(tmp_path):
    path = tmp_path / "marine.csv"
    df = pd.DataFrame(
        {
            "STATION": ["A", "A", "A", "A"],
            "DATE": [
                "2023-01-01 00:00:00",
                "2023-01-01 00:30:00",
                "2023-01-01 01:00:00",
                "2023-01-01 01:30:00",
            ],
            "WIND_SPEED": [2.0, 4.0, 6.0, 8.0],
        }
    )
    df.to_csv(path, index=False)

    inp = read_csv_dataset(path, "DATE", "STATION", expected_columns=["WIND_SPEED"])
    out = resample_by_station(
        inp,
        station_col="STATION",
        timestamp_col="DATE",
        numeric_columns=["WIND_SPEED"],
        rule="1H",
    )

    assert len(out) == 2
    assert out["WIND_SPEED"].iloc[0] == 3.0
    assert out["WIND_SPEED"].iloc[1] == 7.0
