from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "consumption_2021_2024_eda.parquet"
)

df = pd.read_parquet(DATA_PATH)

print("Start:", df["TimeDK"].min())
print("End:", df["TimeDK"].max())

# how does consumption change over time
monthly = (
    df.groupby(df["TimeDK"].dt.to_period("M"))["ConsumptionkWh"]
    .sum()
    .reset_index()
)

monthly["TimeDK"] = monthly["TimeDK"].dt.to_timestamp()

print("Monthly Consumption:")
print(monthly.head())
# plot it
plt.figure(figsize=(14, 5))

plt.plot(
    monthly["TimeDK"],
    monthly["ConsumptionkWh"],
)

plt.title("Monthly electricity consumption, 2021–2024")
plt.xlabel("Month")
plt.ylabel("Consumption (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# compare yearly consumption
annual = (
    df.groupby("year")["ConsumptionkWh"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    annual["year"].astype(str),
    annual["ConsumptionkWh"],
)

plt.title("Annual electricity consumption")
plt.xlabel("Year")
plt.ylabel("Consumption (kWh)")

plt.tight_layout()
plt.show()

# Compare seasonal patterns across years
monthly_by_year = (
    df.groupby(["year", "month"])["ConsumptionkWh"]
    .sum()
    .reset_index()
)

print(monthly_by_year.head())

pivot_month = monthly_by_year.pivot(
    index="month",
    columns="year",
    values="ConsumptionkWh",
)

print(pivot_month)
plt.figure(figsize=(12, 6))

for year in pivot_month.columns:
    plt.plot(
        pivot_month.index,
        pivot_month[year],
        marker="o",
        label=str(year),
    )

plt.title("Monthly electricity consumption by year")
plt.xlabel("Month")
plt.ylabel("Consumption (kWh)")
plt.xticks(range(1, 13))
plt.legend()

plt.tight_layout()
plt.show()