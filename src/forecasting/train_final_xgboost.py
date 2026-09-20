"""Train final regional XGBoost demand models and evaluate on 2025.

Training period: 2021–2024
Test period: 2025
Forecast targets: total regional demand over the next 24 and 168 hours.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "consumption_2021_2025_features.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_2025_metrics.csv"
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_2025_forecasts.csv"
HORIZONS = (24, 168)


def smape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    denominator = (actual.abs() + forecast.abs()).replace(0, np.nan)
    return 2 * (actual - forecast).abs() / denominator * 100


def load_regional_series() -> pd.DataFrame:
    data = pd.read_parquet(
        FEATURE_PATH,
        columns=["TimeUTC", "TimeDK", "RegionName", "ConsumptionkWh"],
    )
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])
    return (
        data.groupby(["RegionName", "TimeUTC", "TimeDK"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values(["RegionName", "TimeUTC"])
    )


def make_supervised(region_data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frame = region_data.sort_values("TimeUTC").copy()
    local = frame["TimeDK"]
    frame["hour"] = local.dt.hour
    frame["weekday"] = local.dt.dayofweek
    frame["month"] = local.dt.month
    frame["day_of_year"] = local.dt.dayofyear
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)

    for column, period in (("hour", 24), ("weekday", 7), ("month", 12), ("day_of_year", 365.25)):
        frame[f"{column}_sin"] = np.sin(2 * np.pi * frame[column] / period)
        frame[f"{column}_cos"] = np.cos(2 * np.pi * frame[column] / period)

    grouped = frame.groupby("RegionName", sort=False)["ConsumptionkWh"]
    for lag in (1, 24, 168):
        frame[f"lag_{lag}h"] = grouped.shift(lag)
    for window in (24, 168):
        frame[f"rolling_mean_{window}h"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )

    frame["future_total"] = grouped.transform(
        lambda values: values.shift(-1).rolling(horizon, min_periods=horizon).sum()
    )
    frame["target_end_year"] = frame["TimeDK"].shift(-horizon).dt.year
    features = [
        "hour", "weekday", "month", "day_of_year", "is_weekend",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_mean_168h",
    ]
    return frame.dropna(subset=features + ["future_total", "target_end_year"]), features


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, horizon), group in predictions.groupby(["region", "horizon_hours"], sort=True):
        rows.append({
            "region": region,
            "horizon_hours": horizon,
            "observations": len(group),
            "mae_kwh": group["absolute_error_kwh"].mean(),
            "rmse_kwh": np.sqrt(group["squared_error_kwh"].mean()),
            "smape_pct": group["smape_pct"].mean(),
        })

    regional = pd.DataFrame(rows)
    all_rows = []
    for horizon, group in predictions.groupby("horizon_hours", sort=True):
        all_rows.append({
            "region": "ALL_REGIONS",
            "horizon_hours": horizon,
            "observations": len(group),
            "mae_kwh": group["absolute_error_kwh"].mean(),
            "rmse_kwh": np.sqrt(group["squared_error_kwh"].mean()),
            "smape_pct": group["smape_pct"].mean(),
        })
    return pd.concat([regional, pd.DataFrame(all_rows)], ignore_index=True)


def main() -> None:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install requirements-forecasting.txt to use XGBoost.") from error

    regional = load_regional_series()
    prediction_rows = []

    for region, values in regional.groupby("RegionName", sort=True):
        for horizon in HORIZONS:
            supervised, features = make_supervised(values, horizon)
            train = supervised[supervised["target_end_year"] <= 2024]
            test = supervised[
                (supervised["target_end_year"] == 2025)
                & (supervised["TimeUTC"].dt.hour == 0)
            ]

            model = XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="mae",
                tree_method="hist",
                n_jobs=4,
                random_state=42,
            )
            model.fit(train[features], train["future_total"])
            forecast = model.predict(test[features])
            actual = test["future_total"].reset_index(drop=True)
            forecast = pd.Series(forecast)
            errors = actual - forecast
            prediction_rows.append(pd.DataFrame({
                "region": region,
                "forecast_origin": test["TimeDK"].reset_index(drop=True),
                "horizon_hours": horizon,
                "actual_total_kwh": actual,
                "forecast_total_kwh": forecast,
                "absolute_error_kwh": errors.abs(),
                "squared_error_kwh": errors ** 2,
                "smape_pct": smape(actual, forecast),
            }))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = summarize(predictions)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(FORECAST_PATH, index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved forecasts to {FORECAST_PATH}")


if __name__ == "__main__":
    main()
