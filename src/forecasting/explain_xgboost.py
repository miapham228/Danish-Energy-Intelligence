"""Create feature-importance summaries for the final regional XGBoost models.

The script retrains the same final models used by ``train_final_xgboost`` and
exports normalized gain-based importance for each region and horizon. SHAP is
deliberately optional; native XGBoost importance keeps the reporting workflow
lightweight and reproducible.
"""

from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support both ``python -m src.forecasting.explain_xgboost`` and direct
# execution via ``python src/forecasting/explain_xgboost.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.train_final_xgboost import HORIZONS, load_regional_series, make_supervised


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_feature_importance.csv"
PLOT_PATH = PROJECT_ROOT / "reports" / "feature_importance_top5.png"


def train_importance(region_data: pd.DataFrame) -> pd.DataFrame:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install requirements-forecasting.txt to use XGBoost.") from error

    rows = []
    for region, values in region_data.groupby("RegionName", sort=True):
        for horizon in HORIZONS:
            supervised, features = make_supervised(values, horizon)
            train = supervised[supervised["target_end_year"] <= 2024]
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
            importances = pd.Series(model.feature_importances_, index=features)
            total = importances.sum()
            for feature, importance in importances.sort_values(ascending=False).items():
                rows.append(
                    {
                        "region": region,
                        "horizon_hours": horizon,
                        "feature": feature,
                        "importance": float(importance),
                        "importance_pct": float(importance / total * 100) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_top_features(importance: pd.DataFrame, output_path: Path = PLOT_PATH) -> None:
    """Plot the five most important average features for each horizon."""
    summary = (
        importance.groupby(["horizon_hours", "feature"], as_index=False)["importance_pct"]
        .mean()
    )
    horizons = sorted(summary["horizon_hours"].unique())
    figure, axes = plt.subplots(1, len(horizons), figsize=(13, 5), squeeze=False)

    for axis, horizon in zip(axes[0], horizons):
        top = (
            summary[summary["horizon_hours"] == horizon]
            .nlargest(5, "importance_pct")
            .sort_values("importance_pct")
        )
        axis.barh(top["feature"], top["importance_pct"], color="#176b87")
        axis.set_title(f"Top 5 features — {horizon}-hour forecast")
        axis.set_xlabel("Mean importance (%)")
        axis.grid(axis="x", alpha=0.25)

    figure.suptitle("XGBoost feature importance by forecast horizon", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    importance = train_importance(load_regional_series())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_PATH, index=False)
    plot_top_features(importance)
    print(importance.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nSaved feature importance to {OUTPUT_PATH}")
    print(f"Saved top-five feature plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
