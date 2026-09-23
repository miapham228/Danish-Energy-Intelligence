import pandas as pd


def test_api_records_can_be_normalized_without_network_access():
    response_payload = {
        "records": [
            {
                "HourUTC": "2025-01-01T00:00:00",
                "PriceArea": "DK1",
                "ConsumptionkWh": 123.4,
            }
        ]
    }

    frame = pd.DataFrame(response_payload["records"])

    assert len(frame) == 1
    assert frame.loc[0, "PriceArea"] == "DK1"
    assert frame["ConsumptionkWh"].dtype.kind in "fi"
