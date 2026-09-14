from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


EXPECTED_COLUMNS = [
    "TimeUTC",
    "TimeDK",
    "RegionName",
    "ConsumerCategory3",
    "ConsumerCategory2",
    "ConsumptionkWh",
]


files = sorted(RAW_DATA_DIR.glob("consumption_2021_*.parquet"))

print(f"Found {len(files)} files.")

all_data = []

for file in files:
    print(f"\nChecking: {file.name}")

    df = pd.read_parquet(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # Check columns
    assert list(df.columns) == EXPECTED_COLUMNS

    # Check missing values
    assert not df.isna().any().any()

    # Check numeric consumption
    assert pd.api.types.is_numeric_dtype(
        df["ConsumptionkWh"]
    )

    # Check negative values
    assert not (df["ConsumptionkWh"] < 0).any()

    # Check duplicate rows
    assert df.duplicated().sum() == 0

    all_data.append(df)


combined = pd.concat(all_data, ignore_index=True)

print("\n==============================")
print("COMBINED DATA")
print("==============================")

print("Shape:")
print(combined.shape)

print("\nTime range:")
print(combined["TimeDK"].min())
print(combined["TimeDK"].max())

print("\nRegions:")
print(combined["RegionName"].unique())

print("\nConsumer categories:")
print(combined["ConsumerCategory3"].unique())

print("\nMissing values:")
print(combined.isna().sum())

print("\nDuplicate rows:")
print(combined.duplicated().sum())

print("\nConsumption summary:")
print(combined["ConsumptionkWh"].describe())