# Prediction vs Actual Tracking & Dashboard

## Problem
We generate daily price predictions (30/60/90 day horizons) but have no way to track how accurate they were. The user needs to see:
1. Historical predictions alongside what actually happened
2. Accuracy metrics over time (are we getting better or worse?)
3. Weekly granularity (1/2/3/4 weeks) in addition to monthly horizons

## Requirements

### R1: Weekly Prediction Horizons
- Add 7, 14, 21, 28 day prediction horizons alongside existing 30, 60, 90
- Generate predictions for all horizons in the daily pipeline

### R2: Prediction History Log
- Every day's predictions saved to `prediction_history.csv`
- Columns: date_generated, horizon_days, target_date, predicted_price, actual_price, error, pct_error
- When target_date arrives, fill in actual_price from hal_prices.csv

### R3: Dashboard — Tahmin Takibi Tab
- New tab in "Tahminler & Uyarılar" page: "Tahmin Takibi" (Prediction Tracking)
- **Prediction vs Actual chart**: lines showing predicted price vs what actually happened
- **Error over time**: rolling MAE/MAPE per horizon
- **Accuracy table**: summary stats per horizon (MAE, MAPE, direction accuracy, interval hit rate)
- **Data freshness**: show when each data source was last updated

### R4: Automated Daily Flow
- `auto_refresh --full` logs predictions and evaluates past ones automatically
- No manual intervention needed

## Out of Scope
- Changing the model architecture
- Adding new data sources
- Alert system changes
