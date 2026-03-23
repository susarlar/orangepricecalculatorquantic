"""
Feature engineering from satellite/NDVI data.

Creates vegetation health indicators, anomaly detection,
and trend features from NDVI time series.
"""
import numpy as np
import pandas as pd


def create_ndvi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate NDVI-derived features.

    Args:
        df: NDVI DataFrame with date, ndvi_mean, ndvi_std columns.

    Returns:
        DataFrame with added NDVI feature columns.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # ── NDVI trend ──
    df["ndvi_lag_1"] = df["ndvi_mean"].shift(1)  # previous observation (~16 days)
    df["ndvi_lag_2"] = df["ndvi_mean"].shift(2)
    df["ndvi_lag_3"] = df["ndvi_mean"].shift(3)

    df["ndvi_change_1"] = df["ndvi_mean"] - df["ndvi_lag_1"]
    df["ndvi_change_2"] = df["ndvi_mean"] - df["ndvi_lag_2"]

    # ── Rolling NDVI statistics ──
    # Note: NDVI observations are ~16 days apart, so window=6 ≈ 3 months
    df["ndvi_roll_mean_3m"] = df["ndvi_mean"].rolling(6, min_periods=2).mean()
    df["ndvi_roll_mean_6m"] = df["ndvi_mean"].rolling(12, min_periods=4).mean()
    df["ndvi_roll_std_3m"] = df["ndvi_mean"].rolling(6, min_periods=2).std()

    # ── Seasonal anomaly ──
    df["month"] = df["date"].dt.month
    monthly_normal = df.groupby("month")["ndvi_mean"].transform("mean")
    df["ndvi_anomaly"] = df["ndvi_mean"] - monthly_normal
    df["ndvi_anomaly_pct"] = (df["ndvi_anomaly"] / monthly_normal) * 100

    # ── Vegetation stress indicators ──
    df["ndvi_below_normal"] = (df["ndvi_anomaly"] < 0).astype(int)
    df["ndvi_stress"] = (df["ndvi_anomaly_pct"] < -15).astype(int)  # >15% below normal

    # ── Year-over-year comparison ──
    # ~23 observations per year (365/16)
    df["ndvi_yoy"] = df["ndvi_mean"] - df["ndvi_mean"].shift(23)
    df["ndvi_yoy_pct"] = (df["ndvi_yoy"] / df["ndvi_mean"].shift(23)) * 100

    # ── Health classification ──
    df["vegetation_health"] = pd.cut(
        df["ndvi_mean"],
        bins=[-1, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0],
        labels=["bare", "very_stressed", "stressed", "moderate", "healthy", "very_healthy"],
    )

    return df


def interpolate_ndvi_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate 16-day NDVI to daily frequency for merging with price data.

    Args:
        df: NDVI DataFrame with date and ndvi_mean.

    Returns:
        Daily DataFrame with interpolated NDVI values.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Resample to daily and interpolate
    daily = df[["ndvi_mean", "ndvi_std"]].resample("D").interpolate(method="linear")
    daily = daily.reset_index()

    return daily
