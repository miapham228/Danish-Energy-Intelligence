import time
import requests
import certifi
import pandas as pd


url = "https://api.energidataservice.dk/dataset/ConsumptionConsumerCategoryHour"

params = {
    "start": "2025-01-01",
    "end": "2025-01-02",
}

headers = {
    "User-Agent": "Danish-Energy-Intelligence/0.1"
}


for attempt in range(3):

    response = requests.get(
        url,
        params=params,
        headers=headers,
        verify=certifi.where(),
        timeout=30,
    )

    print("Status:", response.status_code)

    if response.status_code == 429:
        wait_time = 10 * (attempt + 1)
        print(f"Rate limited. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        continue

    response.raise_for_status()
    break


data = response.json()

print("Keys:", data.keys())
print("Number of records:", len(data["records"]))
print("First record:")
print(data["records"][0])


df = pd.DataFrame(data["records"])

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nMissing values:")
print(df.isna().sum())
