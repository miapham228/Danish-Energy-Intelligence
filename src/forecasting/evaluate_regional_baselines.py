"""Evaluate regional seasonal-naive demand forecasts.

This is the first forecasting benchmark for the project:

    lag 24 hours  -> tomorrow's hourly demand
    lag 168 hours -> next week's hourly demand

The forecast is made separately for each Danish administrative region after
aggregating the three consumer categories. 
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "consumption_2021_2025_features.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "regional_demand_baseline_metrics.csv"


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    denominator = (actual.abs() + forecast.abs()).replace(0, pd.NA)
    return float((2 * (actual - forecast).abs() / denominator).mean() * 100)


def load_regional_hourly_demand() -> pd.DataFrame:
    data = pd.read_parquet(FEATURE_PATH, columns=["TimeUTC", "TimeDK", "RegionName", "ConsumptionkWh"])
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])
    return (
        data.groupby(["RegionName", "TimeUTC", "TimeDK"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values(["RegionName", "TimeUTC"])
    )


def add_baselines(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("RegionName", sort=False)["ConsumptionkWh"]
    data = data.copy()
    data["forecast_tomorrow"] = grouped.shift(24)
    data["forecast_next_week"] = grouped.shift(168)
    data["year"] = data["TimeDK"].dt.year
    return data


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, region_data in data.groupby("RegionName", sort=True):
        for period_name, years in (("validation", [2024]), ("test", [2025])):
            period = region_data[region_data["year"].isin(years)]
            for horizon, prediction_column in (
                ("tomorrow_24h", "forecast_tomorrow"),
                ("next_week_168h", "forecast_next_week"),
            ):
                evaluated = period.dropna(subset=[prediction_column])
                actual = evaluated["ConsumptionkWh"]
                forecast = evaluated[prediction_column]
                rows.append(
                    {
                        "region": region,
                        "period": period_name,
                        "horizon": horizon,
                        "observations": len(evaluated),
                        "mae_kwh": (actual - forecast).abs().mean(),
                        "rmse_kwh": ((actual - forecast) ** 2).mean() ** 0.5,
                        "smape_pct": smape(actual, forecast),
                    }
                )
    return pd.DataFrame(rows)


def plot_test_example(data: pd.DataFrame) -> None:
    """Display the first two weeks of 2025 for each region."""
    test_start = pd.Timestamp("2025-01-01", tz=None)
    test_end = test_start + pd.Timedelta(days=14)
    example = data[(data["TimeDK"] >= test_start) & (data["TimeDK"] < test_end)]

    regions = sorted(data["RegionName"].unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(15, 3 * len(regions)), sharex=True)
    axes = [axes] if len(regions) == 1 else axes
    for axis, region in zip(axes, regions):
        region_data = example[example["RegionName"] == region]
        axis.plot(region_data["TimeDK"], region_data["ConsumptionkWh"], label="Actual", linewidth=1.5)
        axis.plot(region_data["TimeDK"], region_data["forecast_tomorrow"], label="24-hour baseline", alpha=0.8)
        axis.plot(region_data["TimeDK"], region_data["forecast_next_week"], label="168-hour baseline", alpha=0.8)
        axis.set_title(region)
        axis.set_ylabel("kWh")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Danish local time")
    fig.suptitle("Regional demand forecasts: first two weeks of 2025", fontsize=15)
    fig.tight_layout()
    plt.show()


def main() -> None:
    data = add_baselines(load_regional_hourly_demand())
    metrics = evaluate(data)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved metrics to {METRICS_PATH}")
    plot_test_example(data)


if __name__ == "__main__":
    main()
