"""
Weather data pipeline for Finike/Antalya region.

Uses Open-Meteo API (free, no API key required):
- Historical weather data
- Weather forecasts (up to 16 days)
- Climate data
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.config import (
    FINIKE_LAT,
    FINIKE_LON,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_BASE_URL,
    RAW_DIR,
    WEATHER_VARIABLES,
)

logger = logging.getLogger(__name__)


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = FINIKE_LAT,
    longitude: float = FINIKE_LON,
) -> pd.DataFrame:
    """Fetch historical daily weather data from Open-Meteo.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        latitude: Location latitude.
        longitude: Location longitude.

    Returns:
        DataFrame with daily weather variables.
    """
    url = f"{OPEN_METEO_ARCHIVE_URL}/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_VARIABLES),
        "timezone": "Europe/Istanbul",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "daily" not in data:
            logger.warning("No daily data in response")
            return pd.DataFrame()

        daily = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
            "temp_mean": daily.get("temperature_2m_mean"),
            "precipitation": daily.get("precipitation_sum"),
            "humidity": daily.get("relative_humidity_2m_mean"),
            "wind_speed_max": daily.get("wind_speed_10m_max"),
        })

        # Derived features
        df["frost"] = (df["temp_min"] < 0).astype(int)
        df["severe_frost"] = (df["temp_min"] < -5).astype(int)
        df["temp_range"] = df["temp_max"] - df["temp_min"]

        return df

    except Exception as e:
        logger.error(f"Historical weather fetch failed: {e}")
        return pd.DataFrame()


def fetch_weather_forecast(
    latitude: float = FINIKE_LAT,
    longitude: float = FINIKE_LON,
    days: int = 16,
) -> pd.DataFrame:
    """Fetch weather forecast from Open-Meteo (up to 16 days).

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        days: Number of forecast days (max 16).

    Returns:
        DataFrame with forecast weather data.
    """
    url = f"{OPEN_METEO_BASE_URL}/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(WEATHER_VARIABLES),
        "timezone": "Europe/Istanbul",
        "forecast_days": min(days, 16),
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "daily" not in data:
            return pd.DataFrame()

        daily = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
            "temp_mean": daily.get("temperature_2m_mean"),
            "precipitation": daily.get("precipitation_sum"),
            "humidity": daily.get("relative_humidity_2m_mean"),
            "wind_speed_max": daily.get("wind_speed_10m_max"),
        })

        df["frost"] = (df["temp_min"] < 0).astype(int)
        df["severe_frost"] = (df["temp_min"] < -5).astype(int)
        df["temp_range"] = df["temp_max"] - df["temp_min"]
        df["is_forecast"] = True

        return df

    except Exception as e:
        logger.error(f"Forecast fetch failed: {e}")
        return pd.DataFrame()


def fetch_climate_normals(
    latitude: float = FINIKE_LAT,
    longitude: float = FINIKE_LON,
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Calculate climate normals (monthly averages) from historical data.

    Useful for detecting anomalies.
    """
    # Fetch full historical period
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    df = fetch_historical_weather(start_date, end_date, latitude, longitude)
    if df.empty:
        return df

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Monthly normals
    normals = df.groupby("month").agg({
        "temp_max": "mean",
        "temp_min": "mean",
        "temp_mean": "mean",
        "precipitation": "mean",
        "humidity": "mean",
        "frost": "sum",  # average frost days per month
    }).round(2)

    normals.columns = [f"{c}_normal" for c in normals.columns]
    return normals


def compute_weather_anomalies(
    current: pd.DataFrame,
    normals: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weather anomalies vs climate normals.

    Args:
        current: Current weather data with 'date' column.
        normals: Climate normals from fetch_climate_normals().

    Returns:
        DataFrame with anomaly columns added.
    """
    if current.empty or normals.empty:
        return current

    df = current.copy()
    df["month"] = df["date"].dt.month

    # Merge normals
    df = df.merge(normals, left_on="month", right_index=True, how="left")

    # Calculate anomalies
    df["temp_mean_anomaly"] = df["temp_mean"] - df["temp_mean_normal"]
    df["precip_anomaly"] = df["precipitation"] - df["precipitation_normal"]
    df["humidity_anomaly"] = df["humidity"] - df["humidity_normal"]

    return df


def collect_all_weather(start_year: int = 2018) -> pd.DataFrame:
    """Collect all historical weather + current forecast.

    Args:
        start_year: First year of historical data.

    Returns:
        Combined DataFrame with historical + forecast data.
    """
    today = datetime.now()

    # Historical data
    logger.info(f"Fetching historical weather {start_year} to {today.year}")
    historical = fetch_historical_weather(
        start_date=f"{start_year}-01-01",
        end_date=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if not historical.empty:
        historical["is_forecast"] = False

    # Forecast
    logger.info("Fetching 16-day forecast")
    forecast = fetch_weather_forecast()

    # Combine
    parts = [df for df in [historical, forecast] if not df.empty]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="first")
        combined = combined.sort_values("date").reset_index(drop=True)
        return combined

    return pd.DataFrame()


def save_weather(df: pd.DataFrame, filename: str = "weather_finike.csv"):
    """Save weather data to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} weather records to {output_path}")
