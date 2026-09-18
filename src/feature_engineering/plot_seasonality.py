"""Display hourly, weekly, and yearly seasonality diagnostics.

The diagnostic series is the total consumption across all regions and
consumer categories.  It is aggregated by UTC for continuity, while local
Danish time is used for calendar profiles such as hour and weekday.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
YEARS = range(2021, 2026)


def load_hourly_total() -> pd.DataFrame:
    """Load the complete monthly raw files and aggregate to one hourly series."""
    files = [
        RAW_DATA_DIR / f"consumption_{year}_{month:02d}.parquet"
        for year in YEARS
        for month in range(1, 13)
    ]
    missing = [file.name for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError("Missing monthly files: " + ", ".join(missing))

    data = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    data["TimeUTC"] = pd.to_datetime(data["TimeUTC"])
    data["TimeDK"] = pd.to_datetime(data["TimeDK"])

    hourly = (
        data.groupby(["TimeUTC", "TimeDK"], as_index=False)["ConsumptionkWh"]
        .sum()
        .sort_values("TimeUTC")
        .set_index("TimeUTC")
    )
    hourly["hour"] = hourly["TimeDK"].dt.hour
    hourly["weekday"] = hourly["TimeDK"].dt.dayofweek
    hourly["weekday_name"] = hourly["TimeDK"].dt.day_name()
    hourly["month"] = hourly["TimeDK"].dt.month
    hourly["month_name"] = hourly["TimeDK"].dt.month_name()
    hourly["year"] = hourly["TimeDK"].dt.year
    return hourly


def print_lag_correlations(hourly: pd.DataFrame) -> None:
    """Print correlation at common seasonal lags."""
    series = hourly["ConsumptionkWh"]
    print("Lag correlations for the aggregate hourly series:")
    for lag, label in [(24, "24 hours / daily"), (168, "168 hours / weekly")]:
        print(f"  {label:24s}: {series.autocorr(lag=lag):.3f}")

    daily = series.resample("D").sum()
    for lag, label in [(7, "7 days / weekly"), (365, "365 days / yearly")]:
        print(f"  {label:24s}: {daily.autocorr(lag=lag):.3f}")


def plot_diagnostics(hourly: pd.DataFrame) -> None:
    """Display profile, heatmap, and autocorrelation charts."""
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = list(range(1, 13))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Daily/hourly pattern.
    hourly.groupby("hour")["ConsumptionkWh"].mean().plot(
        ax=axes[0, 0], marker="o", color="tab:blue"
    )
    axes[0, 0].set_title("Average load by local hour")
    axes[0, 0].set_xlabel("Hour of day")
    axes[0, 0].set_ylabel("Average consumption (kWh)")

    # 2. Weekly pattern.
    weekday_profile = hourly.groupby("weekday_name")["ConsumptionkWh"].mean().reindex(weekday_order)
    weekday_profile.plot.bar(ax=axes[0, 1], color="tab:green")
    axes[0, 1].set_title("Average load by weekday")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Average consumption (kWh)")
    axes[0, 1].tick_params(axis="x", rotation=35)

    # 3. Hour x weekday interaction.
    heatmap = hourly.pivot_table(
        index="weekday_name", columns="hour", values="ConsumptionkWh", aggfunc="mean"
    ).reindex(weekday_order)
    image = axes[0, 2].imshow(heatmap, aspect="auto", cmap="YlOrRd")
    axes[0, 2].set_title("Average load: weekday × hour")
    axes[0, 2].set_xlabel("Local hour")
    axes[0, 2].set_ylabel("Weekday")
    axes[0, 2].set_yticks(range(7), weekday_order)
    axes[0, 2].set_xticks(range(0, 24, 2))
    fig.colorbar(image, ax=axes[0, 2], label="Average kWh")

    # 4. Yearly/monthly pattern, with one line per calendar year.
    monthly = hourly.groupby(["year", "month"])["ConsumptionkWh"].mean().reset_index()
    for year, group in monthly.groupby("year"):
        axes[1, 0].plot(group["month"], group["ConsumptionkWh"], marker="o", label=str(year))
    axes[1, 0].set_title("Seasonal yearly profile")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Average hourly consumption (kWh)")
    axes[1, 0].set_xticks(month_order)
    axes[1, 0].legend(title="Year")

    # 5. Daily totals across time, useful for seeing annual cycles and trend.
    daily = hourly["ConsumptionkWh"].resample("D").sum()
    axes[1, 1].plot(daily.index, daily.values, color="tab:purple", linewidth=0.8)
    axes[1, 1].set_title("Daily total consumption over time")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Daily consumption (kWh)")

    # 6. Direct evidence for recurring periods.
    candidate_lags = [1, 7, 30, 365, 366]
    correlations = [daily.autocorr(lag=lag) for lag in candidate_lags]
    labels = ["1d", "7d", "30d", "365d", "366d"]
    axes[1, 2].bar(labels, correlations, color="tab:orange")
    axes[1, 2].set_title("Autocorrelation at candidate lags")
    axes[1, 2].set_xlabel("Lag")
    axes[1, 2].set_ylabel("Correlation")
    axes[1, 2].set_ylim(-1, 1)
    axes[1, 2].axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Seasonality diagnostics for Danish electricity consumption", fontsize=16)
    fig.tight_layout()
    plt.show()


def main() -> None:
    hourly = load_hourly_total()
    print(f"Loaded {len(hourly):,} aggregate hourly observations.")
    print(f"Time range: {hourly['TimeDK'].min()} to {hourly['TimeDK'].max()}")
    print_lag_correlations(hourly)
    plot_diagnostics(hourly)


if __name__ == "__main__":
    main()
