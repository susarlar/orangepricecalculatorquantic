"""
Unified feature matrix builder.

Merges all data sources into a single feature matrix
aligned by date for model training.
"""
import logging

import pandas as pd

from src.features.price_features import create_price_features
from src.features.satellite_features import create_ndvi_features, interpolate_ndvi_to_daily
from src.features.weather_features import create_weather_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    prices: pd.DataFrame,
    weather: pd.DataFrame,
    ndvi: pd.DataFrame,
    fx: pd.DataFrame = None,
    foreign_markets: pd.DataFrame = None,
    policy_features: pd.DataFrame = None,
    trends: pd.DataFrame = None,
    demand: pd.DataFrame = None,
    news: pd.DataFrame = None,
    target_horizons: list[int] = None,
) -> pd.DataFrame:
    """Build unified feature matrix from all data sources.

    Args:
        prices: Hal price data.
        weather: Weather data.
        ndvi: NDVI satellite data.
        fx: Foreign exchange rates (optional).
        foreign_markets: FAO index, EU prices, competitor data (optional).
        policy_features: Policy/news event features (optional).
        trends: Google Trends data (optional).
        demand: Demand features — Ramadan, tourism, CPI (optional).
        target_horizons: Prediction horizons in days (default: [30, 60, 90]).

    Returns:
        Feature matrix with target columns for each horizon.
    """
    if target_horizons is None:
        target_horizons = [30, 60, 90]

    # ── Process each data source ──
    logger.info("Creating price features...")
    price_features = create_price_features(prices)
    price_features["date"] = pd.to_datetime(price_features["date"])

    logger.info("Creating weather features...")
    weather_features = create_weather_features(weather)
    weather_features["date"] = pd.to_datetime(weather_features["date"])

    logger.info("Creating NDVI features...")
    ndvi_features = create_ndvi_features(ndvi)
    ndvi_daily = interpolate_ndvi_to_daily(ndvi_features)
    ndvi_daily["date"] = pd.to_datetime(ndvi_daily["date"])

    # ── Merge all features on date ──
    logger.info("Merging feature sources...")
    merged = price_features.copy()

    # Merge weather
    weather_cols = [c for c in weather_features.columns if c != "date" and c not in merged.columns]
    merged = merged.merge(
        weather_features[["date"] + weather_cols],
        on="date",
        how="left",
    )

    # Merge NDVI
    ndvi_cols = [c for c in ndvi_daily.columns if c != "date" and c not in merged.columns]
    merged = merged.merge(
        ndvi_daily[["date"] + ndvi_cols],
        on="date",
        how="left",
    )

    # Merge FX rates
    if fx is not None and not fx.empty:
        fx = fx.copy()
        fx["date"] = pd.to_datetime(fx["date"])
        fx_cols = [c for c in fx.columns if c != "date" and c not in merged.columns]
        merged = merged.merge(
            fx[["date"] + fx_cols],
            on="date",
            how="left",
        )

    # Merge foreign market data (monthly → forward-fill to daily)
    if foreign_markets is not None and not foreign_markets.empty:
        fm = foreign_markets.copy()
        fm["date"] = pd.to_datetime(fm["date"])
        fm_cols = [c for c in fm.columns if c != "date" and c not in merged.columns
                   and c not in ["year", "month"]]
        if fm_cols:
            merged = merged.merge(
                fm[["date"] + fm_cols],
                on="date",
                how="left",
            )
            # Forward-fill monthly data to daily
            for col in fm_cols:
                merged[col] = merged[col].ffill()
            logger.info(f"Merged {len(fm_cols)} foreign market features")

    # Merge policy/news features
    if policy_features is not None and not policy_features.empty:
        pf = policy_features.copy()
        pf["date"] = pd.to_datetime(pf["date"])
        pf_cols = [c for c in pf.columns if c != "date" and c not in merged.columns]
        if pf_cols:
            merged = merged.merge(
                pf[["date"] + pf_cols],
                on="date",
                how="left",
            )
            logger.info(f"Merged {len(pf_cols)} policy/event features")

    # Merge Google Trends (weekly → forward-fill to daily)
    if trends is not None and not trends.empty:
        tr = trends.copy()
        tr["date"] = pd.to_datetime(tr["date"])
        tr_cols = [c for c in tr.columns if c != "date" and c not in merged.columns]
        if tr_cols:
            merged = merged.merge(tr[["date"] + tr_cols], on="date", how="left")
            for col in tr_cols:
                merged[col] = merged[col].ffill()
            logger.info(f"Merged {len(tr_cols)} trend features")

    # Merge demand features (Ramadan, tourism, input costs, CPI)
    if demand is not None and not demand.empty:
        dm = demand.copy()
        dm["date"] = pd.to_datetime(dm["date"])
        dm_cols = [c for c in dm.columns if c != "date" and c not in merged.columns]
        if dm_cols:
            merged = merged.merge(dm[["date"] + dm_cols], on="date", how="left")
            logger.info(f"Merged {len(dm_cols)} demand features")

    # Merge LLM-classified news features (sentiment, volume, per-category counts)
    if news is not None and not news.empty:
        nw = news.copy()
        nw["date"] = pd.to_datetime(nw["date"])
        nw_cols = [c for c in nw.columns if c != "date" and c not in merged.columns]
        if nw_cols:
            merged = merged.merge(nw[["date"] + nw_cols], on="date", how="left")
            # Quiet days = zero, not NaN. Trees handle zeros cleanly.
            for col in nw_cols:
                merged[col] = merged[col].fillna(0)
            logger.info(f"Merged {len(nw_cols)} news features")

    # ── Create target variables ──
    price_col = "avg_price"
    for horizon in target_horizons:
        merged[f"target_{horizon}d"] = merged[price_col].shift(-horizon)
        merged[f"target_{horizon}d_change"] = (
            merged[f"target_{horizon}d"] / merged[price_col] - 1
        )
        merged[f"target_{horizon}d_direction"] = (
            merged[f"target_{horizon}d_change"] > 0
        ).astype(int)

    logger.info(f"Feature matrix: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def get_train_test_split(
    df: pd.DataFrame,
    test_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically for time series validation.

    Args:
        df: Feature matrix with 'date' column.
        test_months: Number of months to hold out for testing.

    Returns:
        (train_df, test_df) tuple.
    """
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].max() - pd.DateOffset(months=test_months)

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    logger.info(f"Train: {len(train)} rows ({train['date'].min()} to {train['date'].max()})")
    logger.info(f"Test:  {len(test)} rows ({test['date'].min()} to {test['date'].max()})")

    return train, test
