"""
Automated daily data refresh, model retraining, and prediction pipeline.

Usage:
    python -m src.auto_refresh              # Refresh data only
    python -m src.auto_refresh --predict    # Refresh + generate predictions
    python -m src.auto_refresh --retrain    # Refresh + rebuild features + retrain models
    python -m src.auto_refresh --full       # Refresh + retrain + predict + alerts (daily prod mode)
    python -m src.auto_refresh --schedule   # Run --full on schedule (every 6 hours)

Exit codes:
    0 = all sources succeeded
    1 = partial failure (some sources failed, pipeline continued)
    2 = critical failure (no price data or model training failed)
"""
import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PREDICTION_HORIZONS, PROCESSED_DIR, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REFRESH_LOG_PATH = PROCESSED_DIR / "refresh_log.csv"


def _log_refresh(source: str, records_before: int, records_after: int,
                 status: str, error_msg: str = ""):
    """Append a row to the refresh log CSV."""
    REFRESH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = REFRESH_LOG_PATH.exists()
    with open(REFRESH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "records_before",
                             "records_after", "status", "error_msg"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source, records_before, records_after, status, error_msg,
        ])


def _count_csv_rows(path: Path) -> int:
    """Count rows in an existing CSV, or return 0."""
    if path.exists():
        try:
            return len(pd.read_csv(path))
        except Exception:
            return 0
    return 0


def refresh_data() -> dict:
    """Fetch latest data from all sources. Returns status dict per source."""
    from src.data.hal_prices import fetch_ibb_monthly, save_prices, PORTAKAL_PRODUCTS
    from src.data.weather import fetch_weather_forecast, fetch_historical_weather, save_weather
    from src.data.fx_rates import fetch_try_rates, save_fx

    logger.info("=" * 50)
    logger.info("DAILY DATA REFRESH")
    logger.info("=" * 50)

    now = datetime.now()
    statuses = {}

    # --- Hal prices ---
    source = "hal_prices"
    prices_path = RAW_DIR / "hal_prices.csv"
    before = _count_csv_rows(prices_path)
    try:
        logger.info("Refreshing hal prices...")
        guid = PORTAKAL_PRODUCTS["portakal"]
        new_prices = []
        # Fetch last 3 months for better coverage
        for month_offset in range(3):
            month = now.month - month_offset
            year = now.year
            if month <= 0:
                month += 12
                year -= 1
            df = fetch_ibb_monthly(year, month, guid)
            if not df.empty:
                new_prices.append(df)

        if prices_path.exists():
            existing = pd.read_csv(prices_path, parse_dates=["date"])
            if new_prices:
                new_df = pd.concat(new_prices, ignore_index=True)
                new_df["date"] = pd.to_datetime(new_df["date"])
                # Only add dates that don't already exist — never overwrite existing data
                existing_dates = set(existing["date"].dt.date)
                new_only = new_df[~new_df["date"].dt.date.isin(existing_dates)]
                if not new_only.empty:
                    combined = pd.concat([existing, new_only], ignore_index=True)
                    combined = combined.sort_values("date").reset_index(drop=True)
                    # Always recalculate avg_price for rows where it's missing
                    mask = combined["avg_price"].isna()
                    combined.loc[mask, "avg_price"] = (
                        combined.loc[mask, "min_price"] + combined.loc[mask, "max_price"]
                    ) / 2
                    save_prices(combined)
                    after = len(combined)
                    logger.info(f"Prices updated: {after} total records (+{after - before} new)")
                    _log_refresh(source, before, after, "ok")
                    statuses[source] = {"status": "ok", "new_rows": after - before}
                else:
                    logger.info("Hal prices: no new dates to add")
                    _log_refresh(source, before, before, "no_new_data")
                    statuses[source] = {"status": "no_new_data", "new_rows": 0}
            else:
                logger.warning("No price data fetched from API")
                _log_refresh(source, before, before, "no_new_data")
                statuses[source] = {"status": "no_new_data", "new_rows": 0}
        else:
            logger.warning("No existing price file found")
            _log_refresh(source, 0, 0, "missing_file")
            statuses[source] = {"status": "missing_file", "new_rows": 0}
    except Exception as e:
        logger.error(f"Hal prices failed: {e}")
        _log_refresh(source, before, before, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    # --- Weather ---
    source = "weather"
    weather_path = RAW_DIR / "weather_finike.csv"
    before = _count_csv_rows(weather_path)
    try:
        logger.info("Refreshing weather forecast...")
        forecast = fetch_weather_forecast()
        if weather_path.exists() and not forecast.empty:
            existing_w = pd.read_csv(weather_path, parse_dates=["date"])
            forecast["is_forecast"] = True
            combined_w = pd.concat([existing_w, forecast], ignore_index=True)
            combined_w = combined_w.drop_duplicates(subset=["date"], keep="first")
            combined_w = combined_w.sort_values("date").reset_index(drop=True)
            save_weather(combined_w)
            after = len(combined_w)
            logger.info(f"Weather updated: {after} records")
            _log_refresh(source, before, after, "ok")
            statuses[source] = {"status": "ok", "new_rows": after - before}
        else:
            _log_refresh(source, before, before, "no_new_data")
            statuses[source] = {"status": "no_new_data", "new_rows": 0}
    except Exception as e:
        logger.error(f"Weather failed: {e}")
        _log_refresh(source, before, before, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    # --- FX rates ---
    source = "fx_rates"
    fx_path = RAW_DIR / "fx_rates.csv"
    before = _count_csv_rows(fx_path)
    try:
        logger.info("Refreshing FX rates...")
        start = (now - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        new_fx = fetch_try_rates(start_date=start)
        if fx_path.exists() and not new_fx.empty:
            existing_fx = pd.read_csv(fx_path, parse_dates=["date"])
            combined_fx = pd.concat([existing_fx, new_fx], ignore_index=True)
            combined_fx = combined_fx.drop_duplicates(subset=["date"], keep="last")
            combined_fx = combined_fx.sort_values("date").reset_index(drop=True)
            save_fx(combined_fx)
            after = len(combined_fx)
            logger.info(f"FX updated: {after} records")
            _log_refresh(source, before, after, "ok")
            statuses[source] = {"status": "ok", "new_rows": after - before}
        else:
            _log_refresh(source, before, before, "no_new_data")
            statuses[source] = {"status": "no_new_data", "new_rows": 0}
    except Exception as e:
        logger.error(f"FX rates failed: {e}")
        _log_refresh(source, before, before, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    # --- Antalya Hal prices (Playwright scraper, optional) ---
    source = "antalya_hal"
    antalya_path = RAW_DIR / "antalya_hal_prices.csv"
    before = _count_csv_rows(antalya_path)
    try:
        logger.info("Refreshing Antalya Hal prices...")
        from src.data.antalya_hal import scrape_antalya_range, save_antalya_prices

        if antalya_path.exists():
            existing_ant = pd.read_csv(antalya_path, parse_dates=["date"])
            last_date = existing_ant["date"].max()
            start_ant = (last_date - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            existing_ant = pd.DataFrame()
            start_ant = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        end_ant = now.strftime("%Y-%m-%d")
        new_ant = scrape_antalya_range(start_ant, end_ant)

        if not new_ant.empty:
            parts = [p for p in (existing_ant, new_ant) if not p.empty]
            combined_ant = pd.concat(parts, ignore_index=True)
            combined_ant["date"] = pd.to_datetime(combined_ant["date"])
            combined_ant = combined_ant.drop_duplicates(subset=["date", "product"], keep="last")
            combined_ant = combined_ant.sort_values(["date", "product"]).reset_index(drop=True)
            save_antalya_prices(combined_ant)
            after = len(combined_ant)
            logger.info(f"Antalya prices updated: {after} total records (+{after - before} new)")
            _log_refresh(source, before, after, "ok")
            statuses[source] = {"status": "ok", "new_rows": after - before}
        else:
            logger.info("Antalya Hal: no new data scraped")
            _log_refresh(source, before, before, "no_new_data")
            statuses[source] = {"status": "no_new_data", "new_rows": 0}
    except ImportError:
        logger.warning("Antalya scraper skipped: playwright not installed")
        _log_refresh(source, before, before, "skipped", "playwright not installed")
        statuses[source] = {"status": "skipped", "error": "playwright missing"}
    except Exception as e:
        logger.warning(f"Antalya Hal failed (non-critical): {e}")
        _log_refresh(source, before, before, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    # --- Demand & policy features ---
    source = "demand_policy"
    try:
        logger.info("Refreshing demand & policy features...")
        from src.data.demand_features import build_demand_features, save_demand_features
        from src.data.policy_events import build_policy_features, save_policy_features

        demand = build_demand_features(start_date="2007-01-01")
        if not demand.empty:
            save_demand_features(demand)
        policy = build_policy_features(start_date="2007-01-01")
        save_policy_features(policy)
        logger.info(f"Demand: {len(demand)} rows, Policy: {len(policy)} rows")
        _log_refresh(source, 0, len(demand) + len(policy), "ok")
        statuses[source] = {"status": "ok"}
    except Exception as e:
        logger.error(f"Demand/policy features failed: {e}")
        _log_refresh(source, 0, 0, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    # --- News (DeepSeek LLM classification) ---
    source = "news_llm"
    try:
        logger.info("Refreshing news + DeepSeek classification...")
        from src.data.news import refresh_news
        news_status = refresh_news()
        st = news_status.get("status", "error")
        relevant = news_status.get("relevant", 0)
        if st == "ok":
            _log_refresh(source, 0, news_status.get("classified", 0), "ok",
                         f"relevant={relevant}")
            logger.info(f"News: {news_status.get('classified', 0)} classified, "
                        f"{relevant} relevant")
        elif st == "no_api_key":
            _log_refresh(source, 0, 0, "skipped", "DEEPSEEK_API_KEY not set")
        elif st == "no_articles":
            _log_refresh(source, 0, 0, "no_new_data", "")
        else:
            _log_refresh(source, 0, 0, "error", news_status.get("error", st))
        statuses[source] = news_status
    except Exception as e:
        logger.warning(f"News classification failed (non-critical): {e}")
        _log_refresh(source, 0, 0, "error", str(e))
        statuses[source] = {"status": "error", "error": str(e)}

    logger.info("Data refresh complete!")
    return statuses


def retrain_models():
    """Rebuild feature matrix and retrain all models."""
    from src.pipeline import build_features, train_models
    from src.models.farmer import train_all_farmer_models

    logger.info("=" * 50)
    logger.info("MODEL RETRAINING")
    logger.info("=" * 50)

    logger.info("Building feature matrix...")
    build_features()

    logger.info("Training baseline models...")
    train_models()

    logger.info("Training farmer models...")
    train_all_farmer_models()

    logger.info("Model retraining complete!")


def should_retrain(statuses: dict) -> bool:
    """Check if retraining is needed: new price data OR missing model files."""
    # Always retrain if model files are missing for any horizon
    for horizon in PREDICTION_HORIZONS:
        model_path = MODELS_DIR / f"xgboost_tuned_{horizon}d.joblib"
        if not model_path.exists():
            logger.info(f"Model missing for {horizon}d horizon — triggering retrain")
            return True

    hal = statuses.get("hal_prices", {})
    if hal.get("status") == "ok" and hal.get("new_rows", 0) > 0:
        return True
    logger.info("No new price data and all models present — skipping retrain")
    return False


def generate_predictions() -> pd.DataFrame:
    """Generate predictions using saved models and full feature matrix."""
    logger.info("Generating predictions...")

    features_path = PROCESSED_DIR / "feature_matrix.csv"
    if not features_path.exists():
        logger.error("Feature matrix not found. Run with --retrain first.")
        return pd.DataFrame()

    features = pd.read_csv(features_path, parse_dates=["date"])
    latest_row = features.iloc[-1:]
    current_price = latest_row["avg_price"].iloc[0]

    results = []
    for horizon in PREDICTION_HORIZONS:
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

    alert_path = PROCESSED_DIR / "latest_alerts.txt"
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(report, encoding="utf-8")

    return alerts


def run_full() -> int:
    """Run the complete daily pipeline. Returns exit code."""
    logger.info("=" * 60)
    logger.info(f"FULL DAILY PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # Step 1: Refresh data
    statuses = refresh_data()

    errors = [s for s in statuses.values() if s.get("status") == "error"]
    if statuses.get("hal_prices", {}).get("status") == "error":
        logger.error("CRITICAL: Price data refresh failed — aborting retrain")
        return 2

    # Step 2: Retrain models (only if new price data)
    if should_retrain(statuses):
        try:
            retrain_models()
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return 2

    # Step 3: Generate predictions & track them
    from src.prediction_tracker import log_predictions, evaluate_predictions, accuracy_report
    preds = generate_predictions()
    if not preds.empty:
        log_predictions(preds)
    evaluate_predictions()
    accuracy_report()

    # Step 4: Run alerts
    try:
        run_alerts()
    except Exception as e:
        logger.warning(f"Alerts failed (non-critical): {e}")

    if errors:
        logger.warning(f"Completed with {len(errors)} source error(s)")
        return 1

    logger.info("Full pipeline completed successfully!")
    return 0


def run_scheduled():
    """Run full pipeline on a schedule using the schedule library."""
    import schedule

    def job():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Scheduled run at {datetime.now()}")
        run_full()

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
    parser.add_argument("--schedule", action="store_true", help="Run --full on schedule (every 6h)")
    parser.add_argument("--predict", action="store_true", help="Refresh + generate predictions")
    parser.add_argument("--retrain", action="store_true", help="Refresh + rebuild features + retrain models")
    parser.add_argument("--full", action="store_true", help="Full daily pipeline: refresh + retrain + predict + alerts")
    parser.add_argument("--alerts", action="store_true", help="Run alerts only")

    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    elif args.full:
        exit_code = run_full()
        sys.exit(exit_code)
    else:
        statuses = refresh_data()

        if args.retrain:
            if should_retrain(statuses):
                retrain_models()
            preds = generate_predictions()
            if not preds.empty:
                from src.prediction_tracker import log_predictions, evaluate_predictions
                log_predictions(preds)
                evaluate_predictions()
            run_alerts()
        elif args.predict:
            preds = generate_predictions()
            if not preds.empty:
                from src.prediction_tracker import log_predictions, evaluate_predictions
                log_predictions(preds)
                evaluate_predictions()
            if args.alerts:
                run_alerts()
        elif args.alerts:
            run_alerts()


if __name__ == "__main__":
    main()
