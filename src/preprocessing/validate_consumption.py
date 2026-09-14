from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "consumption_2025_01.parquet"
)


df = pd.read_parquet(DATA_PATH)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nRegions:")
print(df["RegionName"].unique())

print("\nConsumer categories:")
print(df["ConsumerCategory3"].unique())

print("\nTime range:")
print("Start:", df["TimeDK"].min())
print("End:", df["TimeDK"].max())

print("\nConsumption summary:")
print(df["ConsumptionkWh"].describe())