"""Build plots and a self-contained HTML report for regional forecasts."""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_2025_forecasts.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_2025_metrics.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "regional_xgboost_feature_importance.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "forecast_report"


def figure_data(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def forecast_plot(data: pd.DataFrame, region: str, horizon: int) -> str:
    subset = data[(data["region"] == region) & (data["horizon_hours"] == horizon)].copy()
    subset["forecast_origin"] = pd.to_datetime(subset["forecast_origin"])
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(subset["forecast_origin"], subset["actual_total_kwh"], label="Actual", linewidth=1.2)
    axis.plot(subset["forecast_origin"], subset["forecast_total_kwh"], label="XGBoost", linewidth=1.0)
    axis.set_title(f"{region} — {horizon}-hour demand forecast")
    axis.set_ylabel("Total demand (kWh)")
    axis.set_xlabel("Forecast origin")
    axis.legend()
    axis.grid(alpha=0.25)
    return figure_data(figure)


def importance_plot(importance: pd.DataFrame) -> str | None:
    if importance.empty:
        return None
    summary = (
        importance.groupby(["horizon_hours", "feature"], as_index=False)["importance_pct"]
        .mean()
    )
    top = summary.sort_values(["horizon_hours", "importance_pct"], ascending=[True, False])
    top = top.groupby("horizon_hours", group_keys=False).head(10)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, horizon in zip(axes, sorted(top["horizon_hours"].unique())):
        part = top[top["horizon_hours"] == horizon].sort_values("importance_pct")
        axis.barh(part["feature"], part["importance_pct"])
        axis.set_title(f"Average importance: {horizon} hours")
        axis.set_xlabel("Importance (%)")
    return figure_data(figure)


def build_report(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plots = []
    for region in sorted(forecasts["region"].unique()):
        for horizon in sorted(forecasts["horizon_hours"].unique()):
            plots.append(
                (region, horizon, forecast_plot(forecasts, region, horizon))
            )

    importance_image = importance_plot(importance)
    metric_table = metrics.to_html(index=False, float_format=lambda value: f"{value:,.2f}")
    sections = "\n".join(
        f"<h3>{region} — {horizon}-hour horizon</h3><img src='data:image/png;base64,{image}' />"
        for region, horizon, image in plots
    )
    importance_section = (
        f"<h2>Feature importance</h2><p>Average native XGBoost gain importance across regions.</p>"
        f"<img src='data:image/png;base64,{importance_image}' />"
        if importance_image
        else "<p>Feature-importance output was not found. Run explain_xgboost first.</p>"
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Danish Energy Forecast Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;line-height:1.4}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.4rem;text-align:right}}
th{{background:#f0f3f5}}img{{max-width:100%;margin-bottom:1.5rem}}</style></head>
<body><h1>Danish Energy Forecast Report</h1>
<p>Final XGBoost forecasts evaluated on 2025. Forecasts are regional totals for
the next 24 or 168 hours.</p><h2>Accuracy summary</h2>{metric_table}
{importance_section}<h2>Actual versus predicted demand</h2>{sections}</body></html>"""
    output = REPORT_DIR / "report.html"
    output.write_text(html, encoding="utf-8")
    return str(output)


def main() -> None:
    forecasts = pd.read_csv(FORECAST_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH) if IMPORTANCE_PATH.exists() else pd.DataFrame()
    print(f"Saved report to {build_report(forecasts, metrics, importance)}")


if __name__ == "__main__":
    main()
