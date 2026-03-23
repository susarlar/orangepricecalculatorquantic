"""
Generate realistic synthetic orange price data for model development.

Based on known patterns of Turkish orange (portakal) prices:
- Strong seasonality: low during harvest (Dec-Mar), high off-season (Jul-Oct)
- Year-over-year inflation trend (TRY)
- Frost events cause price spikes
- Random volatility

This is used for development until real hal data APIs are connected.
Real data sources to integrate:
- hal.gov.tr (requires Selenium for SharePoint scraping)
- IBB tarim.ibb.istanbul (daily endpoint, needs browser session)
- CollectAPI (current prices only, needs API key)
"""
import logging

import numpy as np
import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_prices(
    start_date: str = "2018-01-01",
    end_date: str = "2026-03-20",
    base_price_2018: float = 2.5,  # TL/kg in 2018
    annual_inflation: float = 0.45,  # ~45% annual (TRY inflation)
) -> pd.DataFrame:
    """Generate realistic daily orange price data.

    Patterns modeled:
    1. Seasonal cycle: prices lowest during peak harvest (Dec-Feb),
       highest during off-season (Aug-Oct)
    2. Inflation trend: prices rise ~40-50% per year due to TRY inflation
    3. Frost spikes: random frost events cause 20-50% price jumps
    4. Weekly patterns: no trading on Sundays
    5. Random noise: daily price fluctuations

    Args:
        start_date: Start date.
        end_date: End date.
        base_price_2018: Starting price in TL/kg.
        annual_inflation: Annual price increase rate.

    Returns:
        DataFrame with daily price data.
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    for date in dates:
        # Skip Sundays (hal closed)
        if date.dayofweek == 6:
            continue

        # ── Base price with inflation ──
        years_elapsed = (date - pd.Timestamp("2018-01-01")).days / 365.25
        inflation_factor = (1 + annual_inflation) ** years_elapsed
        base = base_price_2018 * inflation_factor

        # ── Seasonal component ──
        # Prices lowest in Dec-Feb (harvest peak), highest in Aug-Oct (off-season)
        day_of_year = date.dayofyear
        # Cosine curve: peak around day 250 (Sep), trough around day 60 (Mar)
        seasonal = 0.20 * np.cos(2 * np.pi * (day_of_year - 250) / 365)

        # ── Frost spike (random events in Dec-Feb) ──
        frost_spike = 0.0
        if date.month in [12, 1, 2]:
            if np.random.random() < 0.02:  # ~2% chance per day in winter
                frost_spike = np.random.uniform(0.20, 0.50)
                # Frost effect persists for ~3-4 weeks (handled by momentum)

        # ── Supply/demand shocks ──
        shock = 0.0
        if np.random.random() < 0.005:  # ~0.5% chance of random shock
            shock = np.random.uniform(-0.15, 0.25)

        # ── Random daily noise ──
        noise = np.random.normal(0, 0.03)

        # ── Combine ──
        price_factor = 1 + seasonal + frost_spike + shock + noise
        avg_price = base * max(price_factor, 0.5)  # floor at 50% of base

        # Min/max spread (typically 10-30% spread in hal)
        spread_pct = np.random.uniform(0.10, 0.30)
        min_price = avg_price * (1 - spread_pct / 2)
        max_price = avg_price * (1 + spread_pct / 2)

        rows.append({
            "date": date,
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2),
            "avg_price": round(avg_price, 2),
            "product": "PORTAKAL",
            "market": "finike_synthetic",
        })

    df = pd.DataFrame(rows)

    # ── Add momentum / autocorrelation ──
    # Real prices don't jump randomly — they have momentum
    df["avg_price"] = df["avg_price"].ewm(span=5).mean().round(2)
    df["min_price"] = (df["avg_price"] * 0.88).round(2)
    df["max_price"] = (df["avg_price"] * 1.12).round(2)

    logger.info(
        f"Generated {len(df)} synthetic price records "
        f"({df['date'].min().date()} to {df['date'].max().date()})"
    )
    logger.info(
        f"Price range: {df['avg_price'].min():.2f} - {df['avg_price'].max():.2f} TL/kg"
    )

    return df


def save_synthetic_prices(df: pd.DataFrame, filename: str = "hal_prices.csv"):
    """Save synthetic price data."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")
