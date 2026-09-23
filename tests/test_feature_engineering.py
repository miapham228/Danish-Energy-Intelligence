import numpy as np
import pandas as pd

from src.feature_engineering.build_features import add_features, danish_holidays


def make_hourly_data(rows: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    frames = []
    for region, offset in (("Region A", 0), ("Region B", 1000)):
        frames.append(
            pd.DataFrame(
                {
                    "TimeUTC": timestamps,
                    "TimeDK": timestamps,
                    "RegionName": region,
                    "ConsumerCategory3": "Total",
                    "ConsumptionkWh": np.arange(rows, dtype=float) + offset,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_danish_holidays_includes_fixed_and_easter_holidays():
    holidays = danish_holidays(2024)

    assert pd.Timestamp("2024-01-01").date() in holidays
    assert pd.Timestamp("2024-03-28").date() in holidays
    assert pd.Timestamp("2024-04-01").date() in holidays


def test_features_are_grouped_and_rolling_values_do_not_include_current_row():
    result = add_features(make_hourly_data())
    region_a = result[result["RegionName"] == "Region A"].reset_index(drop=True)

    assert pd.isna(region_a.loc[0, "consumption_lag_1h"])
    assert region_a.loc[1, "consumption_lag_1h"] == 0
    assert pd.isna(region_a.loc[167, "rolling_mean_168h"])
    assert region_a.loc[168, "rolling_mean_168h"] == np.mean(np.arange(168))
    assert region_a.loc[168, "rolling_mean_168h"] != region_a.loc[168, "ConsumptionkWh"]
