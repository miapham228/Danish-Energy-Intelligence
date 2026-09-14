import pandas as pd

from src.ingestion.energinet import get_consumption_data


import pandas as pd

from src.ingestion.energinet import get_consumption_data


def test_get_consumption_data(monkeypatch):
    fake_response = {
        "records": [
            {
                "TimeUTC": "2025-01-01T00:00:00",
                "TimeDK": "2025-01-01T01:00:00",
                "RegionName": "Region Hovedstaden",
                "ConsumerCategory3": "Erhverv",
                "ConsumerCategory2": "Erhverv",
                "ConsumptionkWh": 1000.5,
            },
            {
                "TimeUTC": "2025-01-01T00:00:00",
                "TimeDK": "2025-01-01T01:00:00",
                "RegionName": "Region Midtjylland",
                "ConsumerCategory3": "Privat",
                "ConsumerCategory2": "Privat",
                "ConsumptionkWh": 800.2,
            },
        ]
    }

    class FakeResponse:
        status_code = 200
        
        def raise_for_status(self):
            pass

        def json(self):
            return fake_response

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "src.ingestion.energinet.requests.get",
        fake_get,
    )

    df = get_consumption_data(
        start="2025-01-01",
        end="2025-01-02",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_columns = [
        "TimeUTC",
        "TimeDK",
        "RegionName",
        "ConsumerCategory3",
        "ConsumerCategory2",
        "ConsumptionkWh",
    ]

    assert list(df.columns) == expected_columns

    assert pd.api.types.is_numeric_dtype(
        df["ConsumptionkWh"]
    )

    assert df["ConsumptionkWh"].notna().all()