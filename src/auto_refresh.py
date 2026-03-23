"""
Automated daily data refresh and prediction pipeline.

Usage:
    python -m src.auto_refresh              # Run once (daily refresh)
    python -m src.auto_refresh --schedule   # Run on schedule (every 6 hours)
    python -m src.auto_refresh --predict    # Refresh + generate predictions
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def refresh_data():
    """Fetch latest data from all sources."""
    from src.data.hal_prices import fetch_ibb_monthly, save_prices, PORTAKAL_PRODUCTS
    from src.data.weather import fetch_weather_forecast, fetch_historical_weather, save_weather
    from src.data.fx_rates import fetch_try_rates, save_fx

    logger.info("=" * 50)
    logger.info("DAILY DATA REFRESH")
    logger.info("=" * 50)

    now = datetime.now()

    # Refresh hal prices (current month + last month)
    logger.info("Refreshing hal prices...")
    guid = PORTAKAL_PRODUCTS["portakal"]
    new_prices = []
    for month_offset in [0, 1]:
        month = now.month - month_offset
        year = now.year
        if month <= 0:
            month += 12
            year -= 1
        df = fetch_ibb_monthly(year, month, guid)
        if not df.empty:
            new_prices.append(df)

    # Load existing and merge
    prices_path = RAW_DIR / "hal_prices.csv"
    if prices_path.exists():
        existing = pd.read_csv(prices_path, parse_dates=["date"])
        if new_prices:
            new_df = pd.concat(new_prices, ignore_index=True)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            if "avg_price" not in combined.columns:
                combined["avg_price"] = (combined["min_price"] + combined["max_price"]) / 2
            save_prices(combined)
            logger.info(f"Prices updated: {len(combined)} total records")

    # Refresh weather (forecast)
    logger.info("Refreshing weather forecast...")
    forecast = fetch_weather_forecast()
    weather_path = RAW_DIR / "weather_finike.csv"
    if weather_path.exists() and not forecast.empty:
        existing_w = pd.read_csv(weather_path, parse_dates=["date"])
        forecast["is_forecast"] = True
        combined_w = pd.concat([existing_w, forecast], ignore_index=True)
        combined_w = combined_w.drop_duplicates(subset=["date"], keep="first")
        combined_w = combined_w.sort_values("date").reset_index(drop=True)
        save_weather(combined_w)
        logger.info(f"Weather updated: {len(combined_w)} records")

    # Refresh FX rates (last month)
    logger.info("Refreshing FX rates...")
    start = (now - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    new_fx = fetch_try_rates(start_date=start)
    fx_path = RAW_DIR / "fx_rates.csv"
    if fx_path.exists() and not new_fx.empty:
        existing_fx = pd.read_csv(fx_path, parse_dates=["date"])
        combined_fx = pd.concat([existing_fx, new_fx], ignore_index=True)
        combined_fx = combined_fx.drop_duplicates(subset=["date"], keep="last")
        combined_fx = combined_fx.sort_values("date").reset_index(drop=True)
        from src.data.fx_rates import save_fx
        save_fx(combined_fx)
        logger.info(f"FX updated: {len(combined_fx)} records")

    logger.info("Data refresh complete!")


def generate_predictions() -> pd.DataFrame:
    """Generate predictions using saved models and full feature matrix."""
    logger.info("Generating predictions...")

    # Load the pre-built feature matrix (has all features already merged)
    features_path = PROCESSED_DIR / "feature_matrix.csv"
    if not features_path.exists():
        logger.error("Feature matrix not found. Run pipeline --features first.")
        return pd.DataFrame()

    features = pd.read_csv(features_path, parse_dates=["date"])
    latest_row = features.iloc[-1:]
    current_price = latest_row["avg_price"].iloc[0]

    results = []
    for horizon in [30, 60, 90]:
        model_path = MODELS_DIR / f"xgboost_tuned_{horizon}d.joblib"
        quantile_path = MODELS_DIR / f"quantile_{horizon}d.joblib"

        if model_path.exists():
            model_data = joblib.load(model_path)
            model = model_data["model"]
            feature_cols = model_data["feature_cols"]
            use_log = model_data.get("use_log", False)

            available = [c for c in feature_cols if c in latest_row.columns]
            if len(available) < len(feature_cols) * 0.5:
                logger.warning(f"Too few features available for {horizon}d model")
                continue

            X = latest_row[feature_cols].fillna(0).values

            pred = model.predict(X)
            if use_log:
                pred = np.expm1(pred)

            result = {
                "horizon_days": horizon,
                "prediction": float(pred[0]),
                "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "target_date": (datetime.now() + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "current_price": float(current_price),
            }

            # Prediction intervals
            if quantile_path.exists():
                q_data = joblib.load(quantile_path)
                q_cols = q_data["feature_cols"]
                X_q = latest_row[q_cols].fillna(0).values
                result["pred_lower"] = float(q_data["models"]["lower"].predict(X_q)[0])
                result["pred_upper"] = float(q_data["models"]["upper"].predict(X_q)[0])

            results.append(result)
            logger.info(
                f"  {horizon}d: {result['prediction']:.2f} TL/kg "
                f"(current: {current_price:.2f})"
            )

    if results:
        pred_df = pd.DataFrame(results)
        pred_path = PROCESSED_DIR / "latest_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")
        return pred_df

    return pd.DataFrame()


def run_alerts():
    """Run alert system with latest data."""
    from src.alerts.scenario_alerts import run_all_alerts, format_alert_report
    from src.features.weather_features import create_weather_features
    from src.features.satellite_features import create_ndvi_features

    weather = pd.read_csv(RAW_DIR / "weather_finike.csv", parse_dates=["date"])
    ndvi_path = RAW_DIR / "ndvi_finike.csv"
    ndvi = pd.read_csv(ndvi_path, parse_dates=["date"]) if ndvi_path.exists() else pd.DataFrame()
    fx = pd.read_csv(RAW_DIR / "fx_rates.csv", parse_dates=["date"])

    if not weather.empty:
        weather = create_weather_features(weather)
    if not ndvi.empty:
        ndvi = create_ndvi_features(ndvi)

    alerts = run_all_alerts(weather=weather, ndvi=ndvi, fx=fx)
    report = format_alert_report(alerts)
    logger.info(f"\n{report}")

    # Save alerts
    alert_path = PROCESSED_DIR / "latest_alerts.txt"
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(report, encoding="utf-8")

    return alerts


def run_scheduled():
    """Run on a schedule using the schedule library."""
    import schedule

    def job():
        logger.info(f"\n{'='*50}")
        logger.info(f"Scheduled run at {datetime.now()}")
        refresh_data()
        generate_predictions()
        run_alerts()

    # Run immediately
    job()

    # Then every 6 hours
    schedule.every(6).hours.do(job)

    logger.info("Scheduler started — running every 6 hours. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Auto-refresh pipeline")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--alerts", action="store_true", help="Run alerts only")

    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        refresh_data()
        if args.predict:
            generate_predictions()
        if args.alerts or args.predict:
            run_alerts()


if __name__ == "__main__":
    main()
