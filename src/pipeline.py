"""
Main pipeline: collect data → build features → train models → generate alerts.

Usage:
    python -m src.pipeline --collect    # Collect all data
    python -m src.pipeline --train      # Train models
    python -m src.pipeline --predict    # Generate predictions + alerts
    python -m src.pipeline --all        # Run everything
"""
import argparse
import logging
from datetime import datetime

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, PREDICTION_HORIZONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def collect_data(start_year: int = 2007):
    """Collect all raw data from APIs."""
    from src.data.hal_prices import collect_historical_prices, save_prices
    from src.data.weather import collect_all_weather, save_weather
    from src.data.satellite import collect_ndvi_timeseries, save_ndvi
    from src.data.fx_rates import fetch_try_rates, save_fx
    from src.data.foreign_markets import collect_all_foreign_data, save_foreign_markets
    from src.data.policy_events import (
        build_policy_events_df, save_policy_events,
        build_policy_features, save_policy_features,
    )

    logger.info("=" * 60)
    logger.info("DATA COLLECTION")
    logger.info("=" * 60)

    # Hal prices (real IBB data)
    logger.info("\n--- Hal Prices ---")
    prices = collect_historical_prices(start_year=start_year)
    if not prices.empty and prices["avg_price"].sum() > 0:
        save_prices(prices)
        logger.info(f"Collected {len(prices)} REAL price records")
    else:
        logger.error("No price data collected!")

    # Weather
    logger.info("\n--- Weather Data ---")
    weather = collect_all_weather(start_year=start_year)
    if not weather.empty:
        save_weather(weather)
        logger.info(f"Collected {len(weather)} weather records")

    # Satellite NDVI
    logger.info("\n--- Satellite NDVI ---")
    ndvi = collect_ndvi_timeseries(start_year=start_year, use_synthetic=True)
    if not ndvi.empty:
        save_ndvi(ndvi)
        logger.info(f"Collected {len(ndvi)} NDVI records")

    # FX rates
    logger.info("\n--- FX Rates ---")
    fx = fetch_try_rates(start_date=f"{start_year}-01-01")
    if not fx.empty:
        save_fx(fx)
        logger.info(f"Collected {len(fx)} FX records")

    # Foreign markets (FAO index, EU prices, competitor data)
    logger.info("\n--- Foreign Markets ---")
    foreign = collect_all_foreign_data(start_year=start_year)
    if not foreign.empty:
        save_foreign_markets(foreign)
        logger.info(f"Collected {len(foreign)} foreign market records")

    # Policy events & features
    logger.info("\n--- Policy & News Events ---")
    events = build_policy_events_df()
    save_policy_events(events)
    logger.info(f"Encoded {len(events)} policy/news events")

    policy_features = build_policy_features(start_date=f"{start_year}-01-01")
    save_policy_features(policy_features)
    logger.info(f"Built {len(policy_features)} daily policy feature records")

    # Google Trends
    logger.info("\n--- Google Trends ---")
    from src.data.trends import fetch_google_trends, save_trends
    trends = fetch_google_trends()
    if not trends.empty:
        save_trends(trends)
        logger.info(f"Collected {len(trends)} trend records")

    # Demand features (Ramadan, tourism, input costs, CPI)
    logger.info("\n--- Demand Features ---")
    from src.data.demand_features import build_demand_features, save_demand_features
    demand = build_demand_features(start_date=f"{start_year}-01-01")
    if not demand.empty:
        save_demand_features(demand)
        logger.info(f"Built {len(demand)} demand feature records")

    logger.info("\nData collection complete!")


def build_features():
    """Build feature matrix from raw data."""
    from src.features.feature_builder import build_feature_matrix

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Load raw data
    prices = _load_csv("hal_prices.csv")
    weather = _load_csv("weather_finike.csv")
    ndvi = _load_csv("ndvi_finike.csv")
    fx = _load_csv("fx_rates.csv")
    foreign = _load_csv("foreign_markets.csv")
    policy = _load_csv("policy_features.csv")
    trends = _load_csv("google_trends.csv")
    demand = _load_csv("demand_features.csv")

    if prices.empty:
        logger.error("No price data found. Run --collect first.")
        return None

    # Build feature matrix
    features = build_feature_matrix(
        prices=prices,
        weather=weather,
        ndvi=ndvi,
        fx=fx,
        foreign_markets=foreign,
        policy_features=policy,
        trends=trends,
        demand=demand,
        target_horizons=PREDICTION_HORIZONS,
    )

    # Save
    output_path = PROCESSED_DIR / "feature_matrix.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Feature matrix saved: {features.shape[0]} rows, {features.shape[1]} columns")

    return features


def train_models():
    """Train and evaluate all baseline models."""
    from src.models.baseline import run_all_models

    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)

    # Load feature matrix
    features = _load_csv("feature_matrix.csv", processed=True)
    if features.empty:
        logger.error("No feature matrix found. Run --features first.")
        return None

    # Run all models
    results = run_all_models(features, horizons=PREDICTION_HORIZONS)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"\n{results.to_string(index=False)}")

    # Save results
    output_path = PROCESSED_DIR / "model_results.csv"
    results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    return results


def train_advanced():
    """Train advanced models with SHAP explainability."""
    from src.models.advanced import run_advanced_models, compute_shap_values, train_tuned_xgboost

    logger.info("=" * 60)
    logger.info("ADVANCED MODEL TRAINING")
    logger.info("=" * 60)

    features = _load_csv("feature_matrix.csv", processed=True)
    if features.empty:
        logger.error("No feature matrix found. Run --features first.")
        return None

    # Train advanced models
    results = run_advanced_models(features, horizons=PREDICTION_HORIZONS)

    # SHAP analysis on best model (30d)
    logger.info("\nComputing SHAP values...")
    xgb_result = train_tuned_xgboost(features, horizon=30)
    if xgb_result:
        shap_result = compute_shap_values(xgb_result, features)
        shap_result["importance"].to_csv(PROCESSED_DIR / "shap_importance.csv", index=False)
        logger.info("SHAP feature importance saved")

        logger.info("\nTop 15 SHAP features:")
        for _, row in shap_result["importance"].head(15).iterrows():
            logger.info(f"  {row['feature']:40s} {row['shap_importance']:.4f}")

    # Combine with baseline results
    baseline_path = PROCESSED_DIR / "model_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        all_results = pd.concat([baseline, results], ignore_index=True)
    else:
        all_results = results

    all_results.to_csv(PROCESSED_DIR / "model_results.csv", index=False)
    logger.info(f"\nAll results saved ({len(all_results)} entries)")

    return results


def generate_alerts():
    """Generate scenario alerts from latest data."""
    from src.alerts.scenario_alerts import run_all_alerts, format_alert_report
    from src.features.weather_features import create_weather_features
    from src.features.satellite_features import create_ndvi_features

    logger.info("=" * 60)
    logger.info("SCENARIO ALERTS")
    logger.info("=" * 60)

    weather = _load_csv("weather_finike.csv")
    ndvi = _load_csv("ndvi_finike.csv")
    fx = _load_csv("fx_rates.csv")

    # Add features needed for alerts
    if not weather.empty:
        weather["date"] = pd.to_datetime(weather["date"])
        weather = create_weather_features(weather)

    if not ndvi.empty:
        ndvi["date"] = pd.to_datetime(ndvi["date"])
        ndvi = create_ndvi_features(ndvi)

    alerts = run_all_alerts(
        weather=weather,
        ndvi=ndvi,
        fx=fx,
    )

    report = format_alert_report(alerts)
    logger.info(f"\n{report}")

    return alerts


def _load_csv(filename: str, processed: bool = False) -> pd.DataFrame:
    """Load a CSV from raw or processed directory."""
    base_dir = PROCESSED_DIR if processed else RAW_DIR
    path = base_dir / filename
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    logger.warning(f"File not found: {path}")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Orange Price Prediction Pipeline")
    parser.add_argument("--collect", action="store_true", help="Collect raw data")
    parser.add_argument("--features", action="store_true", help="Build feature matrix")
    parser.add_argument("--train", action="store_true", help="Train and evaluate models")
    parser.add_argument("--advanced", action="store_true", help="Train advanced models + SHAP")
    parser.add_argument("--alerts", action="store_true", help="Generate scenario alerts")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--start-year", type=int, default=2007, help="Start year for data")

    args = parser.parse_args()

    if args.all or args.collect:
        collect_data(start_year=args.start_year)

    if args.all or args.features:
        build_features()

    if args.all or args.train:
        train_models()

    if args.advanced:
        train_advanced()

    if args.all or args.alerts:
        generate_alerts()

    if not any([args.collect, args.features, args.train, args.advanced, args.alerts, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
