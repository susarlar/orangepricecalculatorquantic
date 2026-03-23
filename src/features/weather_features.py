"""
Feature engineering from weather data.

Creates frost indicators, drought signals, growing degree days,
and weather anomaly features.
"""
import numpy as np
import pandas as pd

from src.config import ALERT_THRESHOLDS


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate weather-derived features for price prediction.

    Args:
        df: Weather DataFrame with date, temp_min, temp_max, precipitation, etc.

    Returns:
        DataFrame with added weather feature columns.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ── Frost features ──
    df["frost"] = (df["temp_min"] < 0).astype(int)
    df["severe_frost"] = (df["temp_min"] < ALERT_THRESHOLDS["frost_severe"]).astype(int)

    # Rolling frost counts
    for window in [7, 14, 30]:
        df[f"frost_days_{window}d"] = df["frost"].rolling(window, min_periods=1).sum()
        df[f"severe_frost_days_{window}d"] = df["severe_frost"].rolling(window, min_periods=1).sum()

    # Coldest temperature in rolling windows
    for window in [7, 14, 30]:
        df[f"temp_min_roll_{window}d"] = df["temp_min"].rolling(window, min_periods=1).min()

    # ── Heat features ──
    df["heat_stress"] = (df["temp_max"] > 38).astype(int)
    for window in [7, 14, 30]:
        df[f"heat_days_{window}d"] = df["heat_stress"].rolling(window, min_periods=1).sum()

    # ── Growing Degree Days (GDD) ──
    # Base temperature for citrus: ~13°C
    base_temp = 13.0
    df["gdd_daily"] = np.maximum(df["temp_mean"] - base_temp, 0)
    df["gdd_cumulative_30d"] = df["gdd_daily"].rolling(30, min_periods=1).sum()
    df["gdd_cumulative_90d"] = df["gdd_daily"].rolling(90, min_periods=1).sum()

    # ── Precipitation features ──
    for window in [7, 14, 30, 60]:
        df[f"precip_sum_{window}d"] = df["precipitation"].rolling(window, min_periods=1).sum()

    # Consecutive dry days
    df["is_dry"] = (df["precipitation"] < 1.0).astype(int)
    df["consecutive_dry_days"] = _consecutive_count(df["is_dry"])

    # Drought indicator
    df["drought_risk"] = (
        df["consecutive_dry_days"] >= ALERT_THRESHOLDS["drought_days"]
    ).astype(int)

    # ── Humidity features ──
    if "humidity" in df.columns:
        for window in [7, 14, 30]:
            df[f"humidity_mean_{window}d"] = df["humidity"].rolling(window, min_periods=1).mean()

        # Disease pressure (high humidity + warm = fungal risk)
        df["disease_pressure"] = (
            (df["humidity"] > 80) & (df["temp_mean"] > 15)
        ).astype(int)
        df["disease_pressure_14d"] = df["disease_pressure"].rolling(14, min_periods=1).sum()

    # ── Temperature variability ──
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["temp_range_7d_avg"] = df["temp_range"].rolling(7, min_periods=1).mean()

    # Temperature trend
    df["temp_mean_7d"] = df["temp_mean"].rolling(7, min_periods=1).mean()
    df["temp_mean_30d"] = df["temp_mean"].rolling(30, min_periods=1).mean()
    df["temp_trend"] = df["temp_mean_7d"] - df["temp_mean_30d"]

    return df


def _consecutive_count(series: pd.Series) -> pd.Series:
    """Count consecutive True/1 values, resetting on False/0."""
    groups = (series != series.shift()).cumsum()
    result = series.groupby(groups).cumsum()
    return result
