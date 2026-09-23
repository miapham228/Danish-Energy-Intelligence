# Danish Energy Intelligence

Forecast Danish electricity demand by administrative region using historical
Energinet consumption data. The project currently covers five regions and
produces forecasts for the next 24 hours and next 168 hours (one week).

## Current status

The forecasting pipeline is implemented and has been evaluated on 2024
validation data and 2025 test data.

The current final XGBoost results on 2025 are:

| Horizon | Overall sMAPE | Observations |
| --- | ---: | ---: |
| 24 hours | 0.56% | 1,825 |
| 168 hours | 1.00% | 1,825 |

XGBoost currently outperforms the SARIMA comparison, particularly for the
168-hour horizon. These results are model benchmarks, not operational
forecasts: uncertainty intervals, monitoring, and rolling retraining are not
implemented yet.

## Repository structure

```text
data/
  raw/                         Monthly source parquet files
  processed/                   Features, forecasts, and evaluation metrics
notebooks/                     Exploration notebook and exported Python script
src/
  ingestion/                   Energinet data download and API helpers
  preprocessing/               Data validation and EDA dataset creation
  feature_engineering/         Feature generation and seasonality plots
  forecasting/                 Baselines and XGBoost/SARIMA training scripts
tests/                         Unit and API tests
```

## Setup

The project uses Python 3.13 in the current development environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-forecasting.txt
```

The raw and processed parquet files are not recreated by the installation
commands. They must already exist under `data/raw/`, or be obtained through
the ingestion scripts.

## Reproducible workflow

Run commands from the repository root.

1. Validate the downloaded history:

   ```powershell
   python -m src.preprocessing.validate_history
   python -m src.preprocessing.validate_consumption
   ```

2. Build the EDA and feature datasets:

   ```powershell
   python -m src.preprocessing.build_eda_dataset
   python -m src.feature_engineering.build_features
   ```

3. Generate seasonal-naive baseline metrics:

   ```powershell
   python -m src.forecasting.evaluate_regional_baselines
   ```

4. Train and evaluate the XGBoost/SARIMA comparison on 2024:

   ```powershell
   python -m src.forecasting.train_xgboost_sarima
   ```

5. Train the final XGBoost models using 2021–2024 and evaluate on 2025:

   ```powershell
   python -m src.forecasting.train_final_xgboost
   ```

Generated metrics and forecasts are written to `data/processed/`.

## Testing

Run the full test suite with:

```powershell
python -m pytest
```

The unit tests use synthetic data for feature and metric checks, so they do
not require loading the large processed parquet file.

## Forecast design

- Input: hourly consumption grouped by `RegionName`.
- Features: calendar variables, cyclic calendar encodings, 1/24/168-hour
  lags, and leakage-safe 24/168-hour rolling means.
- Target: total regional consumption over the following 24 or 168 hours.
- Validation: training through 2023 and evaluation during 2024.
- Test: training through 2024 and evaluation during 2025.
- Metrics: MAE, RMSE, and symmetric mean absolute percentage error (sMAPE).

## Next planned improvements

- Add rolling time-series cross-validation.
- Compare against stronger seasonal baselines.
- Add forecast uncertainty and model monitoring.
- Add forecast plots and feature-importance explanations.
