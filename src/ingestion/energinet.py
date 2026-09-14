import time
from pathlib import Path

import certifi
import pandas as pd
import requests


API_URL = "https://api.energidataservice.dk/dataset/ConsumptionConsumerCategoryHour"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_consumption_data(start: str, end: str) -> pd.DataFrame:
    """
    Download electricity consumption data from Energinet.

    Parameters
    ----------
    start : str
        Start date/time, inclusive. Example: "2025-01-01"
    end : str
        End date/time, exclusive. Example: "2025-01-08"

    Returns
    -------
    pd.DataFrame
        Consumption data.
    """

    params = {
        "start": start,
        "end": end,
    }

    headers = {
        "User-Agent": "Danish-Energy-Intelligence/0.1"
    }

    for attempt in range(3):
        response = requests.get(
            API_URL,
            params=params,
            headers=headers,
            verify=certifi.where(),
            timeout=30,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                wait_time = int(retry_after)
            else:
                wait_time = 10 * (attempt + 1)

            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        break

    else:
        raise RuntimeError("API rate limit exceeded after 3 attempts.")

    data = response.json()

    return pd.DataFrame(data["records"])


def save_raw_data(df: pd.DataFrame, filename: str) -> Path:
    """
    Save raw consumption data as Parquet.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / filename

    df.to_parquet(output_path, index=False)

    return output_path


if __name__ == "__main__":

    start = "2025-01-01"
    end = "2025-02-01"

    print(f"Downloading data from {start} to {end}...")

    df = get_consumption_data(start, end)

    print(f"Downloaded {len(df):,} rows.")

    output_path = save_raw_data(
        df,
        "consumption_2025_01.parquet",
    )

    print(f"Saved to: {output_path}")