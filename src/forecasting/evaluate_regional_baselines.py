"""Evaluate target-aligned seasonal baselines.

Each forecast is made at one UTC midnight per day and predicts the same
24-hour or 168-hour regional total used by the machine-learning models.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "consumption_2021_2025_features.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "regional_baseline_metrics.csv"
HORIZONS = (24, 168)


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    denominator = (actual.abs() + forecast.abs()).replace(0, np.nan)
    return float((2 * (actual - forecast).abs() / denominator).mean() * 100)


def load_regional_hourly_demand() -> pd.DataFrame:
    columns = ["TimeUTC", "TimeDK", "RegionName", "ConsumptionkWh"]
    data = pd.read_parquet(FEATURE_PATH, columns=columns)
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])
    return (
        data.groupby(["RegionName", "TimeUTC", "TimeDK"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values(["RegionName", "TimeUTC"])
    )


def forecast_total(values: pd.Series, origin: int, horizon: int, lags: tuple[int, ...]) -> float:
    """Forecast a future total using one or more historical hourly patterns."""
    array = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
    offsets = np.arange(1, horizon + 1)[:, None]
    source_positions = origin + offsets - np.asarray(lags)[None, :]
    return float(array[source_positions].mean(axis=1).sum())


def evaluate(data: pd.DataFrame, years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    rows = []
    for region, region_data in data.groupby("RegionName", sort=True):
        values = region_data.sort_values("TimeUTC").reset_index(drop=True)
        for origin, timestamp in enumerate(values["TimeUTC"]):
            if timestamp.hour != 0 or timestamp.year not in years:
                continue
            for horizon in HORIZONS:
                end = origin + horizon + 1
                if end > len(values):
                    continue
                actual = float(values.loc[origin + 1: origin + horizon, "ConsumptionkWh"].sum())
                baselines = {
                    "previous_day": (24,),
                    "previous_week": (168,),
                    "four_week_mean": (168, 336, 504, 672),
                }
                for model, lags in baselines.items():
                    if origin + 1 - max(lags) < 0:
                        continue
                    forecast = forecast_total(values["ConsumptionkWh"], origin, horizon, lags)
                    rows.append(
                        {
                            "region": region,
                            "model": model,
                            "period": "validation" if timestamp.year == 2024 else "test",
                            "evaluation_year": timestamp.year,
                            "horizon_hours": horizon,
                            "observations": 1,
                            "mae_kwh": abs(actual - forecast),
                            "rmse_kwh": abs(actual - forecast),
                            "smape_pct": smape(pd.Series([actual]), pd.Series([forecast])),
                        }
                    )
    detail = pd.DataFrame(rows)
    grouped = detail.groupby(
        ["region", "model", "period", "evaluation_year", "horizon_hours"], as_index=False
    )
    return grouped.agg(
        observations=("observations", "sum"),
        mae_kwh=("mae_kwh", "mean"),
        rmse_kwh=("rmse_kwh", lambda values: np.sqrt(np.mean(values ** 2))),
        smape_pct=("smape_pct", "mean"),
    )


def main() -> None:
    metrics = evaluate(load_regional_hourly_demand())
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
