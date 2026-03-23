"""
Satellite data pipeline for Finike citrus region.

Uses free APIs to get NDVI and vegetation health data:
1. Copernicus Data Space (Sentinel-2) via STAC API
2. Google Earth Engine (alternative)
3. Open-source NDVI from Copernicus Climate Data Store

For MVP, we use pre-computed NDVI statistics via the
Copernicus Data Space Ecosystem API (free registration required).
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import FINIKE_BBOX, RAW_DIR

logger = logging.getLogger(__name__)

# Copernicus Data Space STAC API
CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_SEARCH_URL = f"{CDSE_STAC_URL}/search"

# Finike citrus region polygon (GeoJSON)
FINIKE_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [FINIKE_BBOX["min_lon"], FINIKE_BBOX["min_lat"]],
        [FINIKE_BBOX["max_lon"], FINIKE_BBOX["min_lat"]],
        [FINIKE_BBOX["max_lon"], FINIKE_BBOX["max_lat"]],
        [FINIKE_BBOX["min_lon"], FINIKE_BBOX["max_lat"]],
        [FINIKE_BBOX["min_lon"], FINIKE_BBOX["min_lat"]],
    ]]
}


def search_sentinel2_scenes(
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 30,
    limit: int = 100,
) -> list[dict]:
    """Search for Sentinel-2 scenes covering Finike region.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        max_cloud_cover: Maximum cloud cover percentage.
        limit: Max number of results.

    Returns:
        List of scene metadata dicts.
    """
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": FINIKE_POLYGON,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud_cover}
        },
    }

    try:
        resp = requests.post(CDSE_SEARCH_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        scenes = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            scenes.append({
                "id": feature.get("id"),
                "date": props.get("datetime", "")[:10],
                "cloud_cover": props.get("eo:cloud_cover"),
                "platform": props.get("platform"),
                "assets": list(feature.get("assets", {}).keys()),
            })

        logger.info(f"Found {len(scenes)} Sentinel-2 scenes for {start_date} to {end_date}")
        return scenes

    except Exception as e:
        logger.error(f"Sentinel-2 search failed: {e}")
        return []


def compute_ndvi_from_bands(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI from red and NIR bands.

    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to +1 (healthy vegetation > 0.3)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float))
        ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
    return ndvi


def compute_ndvi_statistics(ndvi: np.ndarray) -> dict:
    """Compute summary statistics for an NDVI array.

    Args:
        ndvi: 2D NDVI array.

    Returns:
        Dict with mean, std, min, max, healthy_pct.
    """
    valid = ndvi[np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)]

    if len(valid) == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "healthy_pct": None, "stressed_pct": None}

    return {
        "ndvi_mean": float(np.mean(valid)),
        "ndvi_std": float(np.std(valid)),
        "ndvi_min": float(np.min(valid)),
        "ndvi_max": float(np.max(valid)),
        "ndvi_median": float(np.median(valid)),
        "healthy_pct": float(np.mean(valid > 0.4) * 100),  # % pixels with good vegetation
        "stressed_pct": float(np.mean((valid > 0) & (valid < 0.2)) * 100),  # sparse/stressed
    }


# ─── Alternative: MODIS NDVI via NASA AppEEARS or GIBS ──────────────────────────


def fetch_modis_ndvi_timeseries(
    start_date: str,
    end_date: str,
    latitude: float = 36.2975,
    longitude: float = 30.1483,
) -> pd.DataFrame:
    """Fetch MODIS NDVI time series from NASA (16-day composites).

    Uses the MODIS MOD13Q1 product (250m resolution, 16-day).
    This is a fallback when Sentinel-2 processing is too heavy.

    Note: For production use, register at https://appeears.earthdatacloud.nasa.gov/
    and use the AppEEARS API for point-based NDVI extraction.
    """
    # Placeholder for NASA AppEEARS API integration
    # For now, generate synthetic NDVI based on seasonal patterns
    logger.warning("Using synthetic NDVI data — replace with real API call")
    return _generate_synthetic_ndvi(start_date, end_date)


def _generate_synthetic_ndvi(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic synthetic NDVI data for development/testing.

    Finike orange NDVI patterns:
    - Peak NDVI (0.5-0.7): Spring after harvest (April-June)
    - Moderate NDVI (0.4-0.5): Growing season (July-October)
    - Lower NDVI (0.3-0.45): Harvest season (November-March) — fruit removal
    - Dips: After frost events, drought stress
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="16D")  # MODIS cadence

    np.random.seed(42)
    rows = []
    for date in dates:
        month = date.month
        # Seasonal NDVI pattern for citrus
        if month in [4, 5, 6]:  # post-harvest flush
            base_ndvi = 0.58
        elif month in [7, 8, 9, 10]:  # growing/fruiting
            base_ndvi = 0.48
        elif month in [11, 12, 1, 2, 3]:  # harvest season
            base_ndvi = 0.40
        else:
            base_ndvi = 0.45

        noise = np.random.normal(0, 0.03)
        ndvi = np.clip(base_ndvi + noise, 0.1, 0.8)

        rows.append({
            "date": date,
            "ndvi_mean": round(ndvi, 4),
            "ndvi_std": round(abs(np.random.normal(0.05, 0.01)), 4),
            "healthy_pct": round(np.clip(ndvi * 120, 0, 100), 1),
            "stressed_pct": round(np.clip((1 - ndvi) * 30, 0, 100), 1),
            "source": "synthetic",
        })

    return pd.DataFrame(rows)


# ─── Collection orchestration ───────────────────────────────────────────────────


def collect_ndvi_timeseries(
    start_year: int = 2018,
    end_year: int = None,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """Collect NDVI time series for Finike region.

    Args:
        start_year: First year.
        end_year: Last year (default: current).
        use_synthetic: If True, use synthetic data for development.

    Returns:
        DataFrame with NDVI time series.
    """
    if end_year is None:
        end_year = datetime.now().year

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    if use_synthetic:
        logger.info("Using synthetic NDVI (set use_synthetic=False for real data)")
        return _generate_synthetic_ndvi(start_date, end_date)

    # Real data: search and process Sentinel-2 scenes
    scenes = search_sentinel2_scenes(start_date, end_date)
    if not scenes:
        logger.warning("No scenes found, falling back to synthetic")
        return _generate_synthetic_ndvi(start_date, end_date)

    # For real processing, you'd download bands and compute NDVI per scene
    # This requires rasterio and significant compute — defer to Phase 2
    logger.info(f"Found {len(scenes)} scenes — real NDVI processing not yet implemented")
    return _generate_synthetic_ndvi(start_date, end_date)


def save_ndvi(df: pd.DataFrame, filename: str = "ndvi_finike.csv"):
    """Save NDVI data to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} NDVI records to {output_path}")
