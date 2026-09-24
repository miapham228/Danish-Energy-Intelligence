import numpy as np
import pandas as pd

from src.forecasting.train_xgboost_sarima import add_all_region_metrics, smape
from src.forecasting.train_final_xgboost import summarize
from src.forecasting.evaluate_regional_baselines import forecast_total


def test_smape_returns_percentage():
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([110.0, 180.0])

    expected = np.mean([2 * 10 / 210, 2 * 20 / 380]) * 100
    assert np.isclose(smape(actual, forecast), expected)


def test_add_all_region_metrics_pools_rmse_correctly():
    metrics = pd.DataFrame(
        [
            {"region": "A", "model": "xgboost", "period": "validation", "horizon_hours": 24,
             "observations": 2, "mae_kwh": 10, "rmse_kwh": 12, "smape_pct": 1},
            {"region": "B", "model": "xgboost", "period": "validation", "horizon_hours": 24,
             "observations": 2, "mae_kwh": 20, "rmse_kwh": 24, "smape_pct": 3},
        ]
    )

    result = add_all_region_metrics(metrics)
    pooled = result[result["region"] == "ALL_REGIONS"].iloc[0]

    assert pooled["observations"] == 4
    assert pooled["mae_kwh"] == 15
    assert np.isclose(pooled["rmse_kwh"], np.sqrt((2 * 12**2 + 2 * 24**2) / 4))


def test_summarize_adds_regional_and_pooled_rows():
    predictions = pd.DataFrame(
        {
            "region": ["A", "A", "B"],
            "horizon_hours": [24, 24, 24],
            "absolute_error_kwh": [1.0, 3.0, 2.0],
            "squared_error_kwh": [1.0, 9.0, 4.0],
            "smape_pct": [1.0, 3.0, 2.0],
        }
    )

    result = summarize(predictions)
    pooled = result[result["region"] == "ALL_REGIONS"].iloc[0]

    assert set(result["region"]) == {"A", "B", "ALL_REGIONS"}
    assert pooled["observations"] == 3
    assert pooled["mae_kwh"] == 2


def test_forecast_total_uses_only_historical_values():
    values = pd.Series(np.arange(1000, dtype=float))

    forecast = forecast_total(values, origin=700, horizon=3, lags=(7,))

    assert forecast == 694 + 695 + 696
