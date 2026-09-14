from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "consumption_2021_2024_eda.parquet"

EXPECTED_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "RegionName",
    "ConsumerCategory3",
    "ConsumerCategory2",
    "ConsumptionkWh",
]


def build_eda_dataset() -> pd.DataFrame:
    """Combine and prepare the 2021–2024 monthly raw files for EDA."""

    files = [
        RAW_DATA_DIR / f"consumption_{year}_{month:02d}.parquet"
        for year in range(2021, 2025)
        for month in range(1, 13)
    ]
    missing_files = [file.name for file in files if not file.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing monthly raw files: " + ", ".join(missing_files)
        )

    monthly_data = []
    for file in files:
        data = pd.read_parquet(file)
        if list(data.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {file.name}")
        data["source_file"] = file.name
        monthly_data.append(data)

    df = pd.concat(monthly_data, ignore_index=True)

    # Normalize the Danish character so the region label is stable across files.
    df["RegionName"] = df["RegionName"].replace(
        {
            "Region Sj\ufffdlland": "Region Sj\u00e6lland",
            "Region Sj\u00e6lland": "Region Sj\u00e6lland",
        }
    )

    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"])
    df["TimeDK"] = pd.to_datetime(df["TimeDK"])

    df["date"] = df["TimeDK"].dt.date
    df["year"] = df["TimeDK"].dt.year
    df["month"] = df["TimeDK"].dt.month
    df["month_name"] = df["TimeDK"].dt.month_name()
    df["day_of_week"] = df["TimeDK"].dt.dayofweek
    df["day_name"] = df["TimeDK"].dt.day_name()
    df["hour"] = df["TimeDK"].dt.hour
    df["is_weekend"] = df["day_of_week"] >= 5

    if df.isna().any().any():
        raise ValueError("Processed data contains missing values")
    if (df["ConsumptionkWh"] < 0).any():
        raise ValueError("Processed data contains negative consumption")

    key_columns = [
        "TimeUTC",
        "RegionName",
        "ConsumerCategory3",
        "ConsumerCategory2",
    ]
    if df.duplicated(key_columns).any():
        raise ValueError("Duplicate time/region/category observations found")

    return df.sort_values(key_columns).reset_index(drop=True)


if __name__ == "__main__":
    processed = build_eda_dataset()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(processed):,} rows to {OUTPUT_PATH}")
    print(f"Columns: {', '.join(processed.columns)}")
    print(f"Time range: {processed['TimeDK'].min()} to {processed['TimeDK'].max()}")
