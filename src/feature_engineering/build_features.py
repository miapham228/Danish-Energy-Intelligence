"""Build leakage-safe forecasting features for electricity consumption.

The features are created independently for each region/consumer-category
series. 

The output retains the initial warm-up rows with missing lag/rolling values.
Those rows should be removed from a model-training set after the split.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "consumption_2021_2025_features.parquet"
YEARS = range(2021, 2026)
SERIES_KEY = ["RegionName", "ConsumerCategory3"]


def load_raw_data() -> pd.DataFrame:
    """Load the complete monthly raw data for 2021–2025."""
    files = [
        RAW_DATA_DIR / f"consumption_{year}_{month:02d}.parquet"
        for year in YEARS
        for month in range(1, 13)
    ]
    missing = [file.name for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError("Missing monthly files: " + ", ".join(missing))

    data = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])
    data["RegionName"] = data["RegionName"].replace(
        {
            "Region Sjï¿½lland": "Region Sj\u00e6lland",
            "Region Sj�lland": "Region Sj\u00e6lland",
            "Region SjÃ¦lland": "Region Sj\u00e6lland",
        }
    )
    return data.sort_values(SERIES_KEY + ["TimeUTC"]).reset_index(drop=True)


def easter_sunday(year: int) -> date:
    """Return Easter Sunday using the Gregorian computus algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def danish_holidays(year: int) -> set[date]:
    """Return major Danish public holidays for one year."""
    easter = easter_sunday(year)
    return {
        date(year, 1, 1),  # New Year's Day
        date(year, 5, 1),  # Labour Day
        date(year, 6, 5),  # Constitution Day
        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 26),
        date(year, 12, 31),
        easter - timedelta(days=3),  # Maundy Thursday
        easter - timedelta(days=2),  # Good Friday
        easter,
        easter + timedelta(days=1),  # Easter Monday
        easter + timedelta(days=26),  # Ascension Day
        easter + timedelta(days=49),  # Whit Monday
    }


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclic, lag, and leakage-safe rolling features."""
    data = data.copy()
    local = data["TimeDK"]

    data["year"] = local.dt.year
    data["month"] = local.dt.month
    data["day_of_month"] = local.dt.day
    data["day_of_year"] = local.dt.dayofyear
    data["hour"] = local.dt.hour
    data["day_of_week"] = local.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype("int8")
    data["week_of_year"] = local.dt.isocalendar().week.astype("int16")
    data["days_since_start"] = (
        (data["TimeUTC"] - data["TimeUTC"].min()).dt.total_seconds() / 86_400
    )

    # Cyclic encodings preserve the circular nature of calendar variables.
    for column, period in [("hour", 24), ("day_of_week", 7), ("month", 12), ("day_of_year", 365.25)]:
        data[f"{column}_sin"] = np.sin(2 * np.pi * data[column] / period)
        data[f"{column}_cos"] = np.cos(2 * np.pi * data[column] / period)

    data["utc_offset_hours"] = (
        data["TimeDK"] - data["TimeUTC"]
    ).dt.total_seconds() / 3_600
    data["is_daylight_saving"] = (data["utc_offset_hours"] == 2).astype("int8")

    holidays = {
        holiday
        for year in YEARS
        for holiday in danish_holidays(year)
    }
    data["is_danish_holiday"] = data["TimeDK"].dt.date.isin(holidays).astype("int8")

    grouped = data.groupby(SERIES_KEY, sort=False)["ConsumptionkWh"]
    for lag in (1, 24, 168):
        data[f"consumption_lag_{lag}h"] = grouped.shift(lag)

    # Shift before rolling so the current target is never included.
    for window, label in ((24, "24h"), (168, "168h")):
        data[f"rolling_mean_{label}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window=window, min_periods=window).mean()
        )
        data[f"rolling_std_{label}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window=window, min_periods=window).std()
        )

    return data


def main() -> None:
    data = add_features(load_raw_data())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)

    lag_columns = ["consumption_lag_1h", "consumption_lag_24h", "consumption_lag_168h"]
    print(f"Saved {len(data):,} rows to {OUTPUT_PATH}")
    print(f"Feature count: {len(data.columns)} columns")
    print("Warm-up rows missing lag features:")
    print(data[lag_columns].isna().sum().to_string())


if __name__ == "__main__":
    main()

import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("data\processed\consumption_2021_2025_features.parquet")

# Read just the first row group / batch instead of the whole file
batch = next(parquet_file.iter_batches(batch_size=5))
df = batch.to_pandas()
print(df)
