"""
Demand-side features for orange price prediction.

Includes:
- Ramadan/holiday calendar (demand spikes)
- Input cost index (fuel, fertilizer, labor)
- Tourism seasonality for Antalya region
- CPI/inflation proxy
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


# Ramadan start dates (approximate, 1st day of Ramadan)
# Moves ~11 days earlier each year in the Gregorian calendar
RAMADAN_DATES = {
    2007: "2007-09-13", 2008: "2008-09-01", 2009: "2009-08-22",
    2010: "2010-08-11", 2011: "2011-08-01", 2012: "2012-07-20",
    2013: "2013-07-09", 2014: "2014-06-29", 2015: "2015-06-18",
    2016: "2016-06-06", 2017: "2017-05-27", 2018: "2018-05-16",
    2019: "2019-05-06", 2020: "2020-04-24", 2021: "2021-04-13",
    2022: "2022-04-02", 2023: "2023-03-23", 2024: "2024-03-12",
    2025: "2025-03-01", 2026: "2026-02-18",
}


def build_demand_features(start_date: str = "2007-01-01", end_date: str = None) -> pd.DataFrame:
    """Build daily demand-side features.

    Args:
        start_date: Start date.
        end_date: End date (default: today).

    Returns:
        Daily DataFrame with demand features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})

    # Ramadan features
    df = _add_ramadan_features(df)

    # Input cost index
    df = _add_input_cost_index(df)

    # Tourism seasonality
    df = _add_tourism_features(df)

    # Inflation/CPI proxy
    df = _add_inflation_proxy(df)

    # Day-of-week demand pattern
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek

    return df


def _add_ramadan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ramadan calendar features.

    Ramadan drives higher fruit consumption (iftar meals).
    Pre-Ramadan buying starts ~1 week before.
    """
    df["ramadan_active"] = 0
    df["pre_ramadan"] = 0
    df["bayram_period"] = 0  # Eid al-Fitr (3 days after Ramadan)

    for year, start_str in RAMADAN_DATES.items():
        start = pd.Timestamp(start_str)
        end = start + pd.Timedelta(days=29)  # Ramadan is ~30 days
        pre_start = start - pd.Timedelta(days=7)
        bayram_end = end + pd.Timedelta(days=3)

        mask_ramadan = (df["date"] >= start) & (df["date"] <= end)
        mask_pre = (df["date"] >= pre_start) & (df["date"] < start)
        mask_bayram = (df["date"] > end) & (df["date"] <= bayram_end)

        df.loc[mask_ramadan, "ramadan_active"] = 1
        df.loc[mask_pre, "pre_ramadan"] = 1
        df.loc[mask_bayram, "bayram_period"] = 1

    # Days until next Ramadan
    df["days_to_ramadan"] = 365  # default far away
    for year, start_str in RAMADAN_DATES.items():
        start = pd.Timestamp(start_str)
        for offset in range(-60, 1):
            target = start + pd.Timedelta(days=offset)
            mask = df["date"] == target
            if mask.any():
                df.loc[mask, "days_to_ramadan"] = min(-offset, df.loc[mask, "days_to_ramadan"].iloc[0])

    return df


def _add_input_cost_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite input cost index.

    Based on known Turkish agricultural input cost trends:
    - Fuel (mazot): strongly correlated with USD/TRY
    - Fertilizer: global commodity + FX
    - Labor: minimum wage increases (Jan each year)
    """
    # Annual composite input cost index (2015=100)
    input_costs = {
        2007: 45, 2008: 52, 2009: 48, 2010: 50,
        2011: 55, 2012: 58, 2013: 62, 2014: 68,
        2015: 100, 2016: 110, 2017: 120, 2018: 155,
        2019: 170, 2020: 195, 2021: 250, 2022: 450,
        2023: 620, 2024: 780, 2025: 920, 2026: 1000,
    }

    df["input_cost_index"] = df["date"].dt.year.map(input_costs)

    # Add monthly variation (fuel price fluctuations)
    month = df["date"].dt.month
    seasonal_adj = 1.0 + (month.isin([1, 2, 3]).astype(float) * 0.05)  # winter fuel costs
    df["input_cost_index"] = df["input_cost_index"] * seasonal_adj

    # Year-over-year input cost change
    df["input_cost_yoy"] = df["input_cost_index"].pct_change(365)

    return df


def _add_tourism_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Antalya tourism seasonality.

    Tourism drives local fruit demand in summer.
    Antalya receives ~15M tourists/year, peaked in summer months.
    """
    month = df["date"].dt.month

    # Relative tourism intensity by month (peak July=1.0)
    tourism_map = {
        1: 0.10, 2: 0.08, 3: 0.15, 4: 0.30,
        5: 0.55, 6: 0.80, 7: 1.00, 8: 0.95,
        9: 0.75, 10: 0.45, 11: 0.15, 12: 0.10,
    }
    df["tourism_intensity"] = month.map(tourism_map)

    # Tourism growing over years
    year_factor = (df["date"].dt.year - 2007) / (2026 - 2007)
    df["tourism_intensity"] = df["tourism_intensity"] * (0.7 + 0.3 * year_factor)

    # COVID dip
    covid_mask = (df["date"] >= "2020-03-15") & (df["date"] <= "2021-06-01")
    df.loc[covid_mask, "tourism_intensity"] *= 0.2

    return df


def _add_inflation_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Add Turkish CPI/inflation proxy.

    Annual CPI (2007=100 base).
    """
    # Approximate Turkish CPI index
    cpi = {
        2007: 100, 2008: 110, 2009: 117, 2010: 127,
        2011: 137, 2012: 149, 2013: 160, 2014: 174,
        2015: 188, 2016: 203, 2017: 226, 2018: 272,
        2019: 313, 2020: 360, 2021: 430, 2022: 740,
        2023: 1140, 2024: 1520, 2025: 1830, 2026: 2050,
    }
    df["cpi_index"] = df["date"].dt.year.map(cpi)
    df["inflation_yoy"] = df["cpi_index"].pct_change(365)

    # Real price = nominal / CPI * 100
    # (will be computed in feature builder when merged with prices)

    return df


def save_demand_features(df: pd.DataFrame, filename: str = "demand_features.csv"):
    """Save demand features to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} demand feature records to {output_path}")
