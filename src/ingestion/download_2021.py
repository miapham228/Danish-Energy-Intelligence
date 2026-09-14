from calendar import monthrange
from pathlib import Path

try:
    from .energinet import get_consumption_data, save_raw_data
except ImportError:
    from energinet import get_consumption_data, save_raw_data


START_YEAR = 2021
END_YEAR = 2026
# September 2026 is still in progress; only download complete months.
LAST_COMPLETE_YEAR = 2026
LAST_COMPLETE_MONTH = 8
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_month(year: int, month: int) -> None:
    """Download one complete calendar month unless its raw file exists."""

    filename = f"consumption_{year}_{month:02d}.parquet"
    output_path = RAW_DATA_DIR / filename

    if output_path.exists():
        print(f"Skipping {year}-{month:02d}: {output_path} already exists.")
        return

    last_day = monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"

    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"

    print(f"Downloading {start} through {year}-{month:02d}-{last_day:02d}...")
    df = get_consumption_data(start=start, end=end)
    save_raw_data(df, filename)
    print(f"Saved {len(df):,} rows to {output_path}.")


if __name__ == "__main__":
    for year in range(START_YEAR, END_YEAR + 1):
        last_month = 12
        if year == LAST_COMPLETE_YEAR:
            last_month = LAST_COMPLETE_MONTH

        for month in range(1, last_month + 1):
            download_month(year, month)
