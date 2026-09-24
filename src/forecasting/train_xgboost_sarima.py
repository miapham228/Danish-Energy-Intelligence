"""Train regional XGBoost and SARIMA demand forecasts.

The target is total regional demand over the next 24 or 168 hours. XGBoost
uses direct supervised features. SARIMA runs on daily regional totals because
the targets are daily and weekly totals; this is much faster than fitting an
hourly SARIMA model. This validation-only script trains on 2021–2023 and
evaluates the complete 2024 validation year.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "consumption_2021_2025_features.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "regional_validation_metrics.csv"
HORIZONS = (24, 168)
VALIDATION_YEARS = (2022, 2023, 2024)
SERIES_KEY = "RegionName"


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    denominator = (actual.abs() + forecast.abs()).replace(0, np.nan)
    return float((2 * (actual - forecast).abs() / denominator).mean() * 100)


def load_regional_series() -> pd.DataFrame:
    columns = ["TimeUTC", "TimeDK", "RegionName", "ConsumptionkWh"]
    data = pd.read_parquet(FEATURE_PATH, columns=columns)
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])
    return (
        data.groupby(["RegionName", "TimeUTC", "TimeDK"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values(["RegionName", "TimeUTC"])
    )


def make_supervised(region_data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Create origin-time features and a future 24/168-hour total target."""
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

    grouped = frame.groupby(SERIES_KEY, sort=False)["ConsumptionkWh"]
    for lag in (1, 24, 168):
        frame[f"lag_{lag}h"] = grouped.shift(lag)

    # Every feature must be known at the forecast origin.
    for window in (24, 168):
        frame[f"rolling_mean_{window}h"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )

    frame["future_total"] = grouped.transform(
        lambda values: values.shift(-1).rolling(horizon, min_periods=horizon).sum()
    )
    frame["target_end_year"] = frame["TimeDK"].shift(-horizon).dt.year
    feature_columns = [
        "hour", "weekday", "month", "day_of_year", "is_weekend",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_mean_168h",
    ]
    usable = frame.dropna(subset=feature_columns + ["future_total", "target_end_year"])
    return usable, usable["future_total"]


def score(
    actual: pd.Series,
    forecast: np.ndarray,
    region: str,
    model: str,
    period: str,
    horizon: int,
    evaluation_year: int | None = None,
) -> dict:
    actual = pd.Series(actual).reset_index(drop=True)
    forecast = pd.Series(forecast).reset_index(drop=True)
    return {
        "region": region,
        "model": model,
        "period": period,
        "evaluation_year": evaluation_year,
        "horizon_hours": horizon,
        "observations": len(actual),
        "mae_kwh": (actual - forecast).abs().mean(),
        "rmse_kwh": np.sqrt(((actual - forecast) ** 2).mean()),
        "smape_pct": smape(actual, forecast),
    }


def add_all_region_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add pooled metrics for model selection across all regions."""
    summary_rows = []
    grouping = ["model", "period", "horizon_hours"]
    if "evaluation_year" in metrics.columns:
        grouping.append("evaluation_year")
    for keys, group in metrics.groupby(grouping, sort=False, dropna=False):
        model, period, horizon = keys[:3]
        observations = group["observations"].sum()
        row = {
                "region": "ALL_REGIONS",
                "model": model,
                "period": period,
                "horizon_hours": horizon,
                "observations": observations,
                "mae_kwh": (group["mae_kwh"] * group["observations"]).sum() / observations,
                "rmse_kwh": np.sqrt(
                    (group["rmse_kwh"] ** 2 * group["observations"]).sum() / observations
                ),
                "smape_pct": (group["smape_pct"] * group["observations"]).sum() / observations,
            }
        if "evaluation_year" in metrics.columns:
            row["evaluation_year"] = keys[3]
        summary_rows.append(row)
    return pd.concat([metrics, pd.DataFrame(summary_rows)], ignore_index=True)


def evaluation_origins(sample: pd.DataFrame) -> pd.DataFrame:
    """Use one forecast origin per UTC day for comparable full-year metrics."""
    return sample[sample["TimeUTC"].dt.hour == 0].copy()


def run_xgboost(region_data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install requirements-forecasting.txt to use XGBoost.") from error

    metrics = []
    forecasts = []
    for region, values in region_data.groupby(SERIES_KEY, sort=True):
        for horizon in HORIZONS:
            supervised, target = make_supervised(values, horizon)
            features = [column for column in supervised.columns if column.startswith(("hour", "weekday", "month", "day_of_year", "is_weekend", "lag_", "rolling_"))]
            for evaluation_year in VALIDATION_YEARS:
                train = supervised[supervised["target_end_year"] <= evaluation_year - 1]
                validation = evaluation_origins(
                    supervised[supervised["target_end_year"] == evaluation_year]
                )
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
                model.fit(train[features], target.loc[train.index])
                prediction = model.predict(validation[features])
                metrics.append(score(
                    validation["future_total"], prediction, region, "xgboost",
                    "validation", horizon, evaluation_year,
                ))
    return metrics, forecasts


def run_sarima(region_data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as error:
        raise RuntimeError("Install requirements-forecasting.txt to use SARIMA.") from error

    metrics = []
    forecasts = []

    for region, values in region_data.groupby(SERIES_KEY, sort=True):
        hourly = values.set_index("TimeUTC")["ConsumptionkWh"].asfreq("h")
        hourly = hourly.interpolate(limit_direction="both")
        daily = hourly.resample("D").sum()

        for future_year in VALIDATION_YEARS:
            cutoff = pd.Timestamp(f"{future_year - 1}-12-31 23:00:00")
            period = "validation"
            history = daily.loc[:cutoff].tail(365)
            model = SARIMAX(
                history,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(method="powell", maxiter=100)

            period_values = daily[daily.index.year == future_year]
            predictions = {horizon: [] for horizon in HORIZONS}
            actuals = {horizon: [] for horizon in HORIZONS}

            # Append observed values one day at a time. At each daily origin,
            # forecast one day or seven days ahead.
            for timestamp, observed in period_values.items():
                model = model.append([observed], refit=False)
                future = daily.loc[timestamp + pd.Timedelta(days=1):]
                if len(future) < 7:
                    continue
                prediction_week = model.forecast(steps=7)
                predictions[24].append(prediction_week.iloc[:1].sum())
                actuals[24].append(future.iloc[:1].sum())
                predictions[168].append(prediction_week.sum())
                actuals[168].append(future.iloc[:7].sum())

            for horizon in HORIZONS:
                metrics.append(score(
                    pd.Series(actuals[horizon]),
                    np.asarray(predictions[horizon]),
                    region,
                    "sarima",
                    period,
                    horizon,
                    future_year,
                ))

    return metrics, forecasts


def main() -> None:
    regional = load_regional_series()
    xgb_metrics, xgb_forecasts = run_xgboost(regional)
    sarima_metrics, sarima_forecasts = run_sarima(regional)
    metrics = add_all_region_metrics(pd.DataFrame(xgb_metrics + sarima_metrics))
    metrics.to_csv(METRICS_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
