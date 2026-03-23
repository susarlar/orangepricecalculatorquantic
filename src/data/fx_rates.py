"""
Foreign exchange rate data pipeline.

Uses free APIs to get currency rates relevant to orange trade.
"""
import logging
from datetime import datetime

import pandas as pd
import requests

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

# Free FX API (no key required)
FRANKFURTER_URL = "https://api.frankfurter.app"


def fetch_fx_history(
    base: str = "TRY",
    symbols: list[str] = None,
    start_date: str = "2018-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch historical FX rates from Frankfurter API.

    Args:
        base: Base currency.
        symbols: Target currencies.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (default: today).

    Returns:
        DataFrame with daily FX rates.
    """
    if symbols is None:
        symbols = ["USD", "EUR", "GBP", "RUB"]
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{FRANKFURTER_URL}/{start_date}..{end_date}"
    params = {"base": base, "to": ",".join(symbols)}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rates = data.get("rates", {})
        rows = []
        for date_str, rate_dict in rates.items():
            row = {"date": pd.to_datetime(date_str)}
            for currency, rate in rate_dict.items():
                row[f"{base}_{currency}"] = rate
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} FX records")
        return df

    except Exception as e:
        logger.error(f"FX fetch failed: {e}")
        return pd.DataFrame()


def fetch_try_rates(start_date: str = "2018-01-01") -> pd.DataFrame:
    """Fetch TRY rates against key trade currencies."""
    # Frankfurter doesn't support TRY as base, so invert from USD
    df = fetch_fx_history(
        base="USD",
        symbols=["TRY", "EUR", "EGP", "MAD", "ZAR", "RUB"],
        start_date=start_date,
    )

    if df.empty:
        return df

    # Calculate cross rates vs TRY
    if "USD_TRY" in df.columns:
        try_rate = df["USD_TRY"]
        for col in df.columns:
            if col.startswith("USD_") and col != "USD_TRY":
                currency = col.split("_")[1]
                df[f"TRY_{currency}"] = try_rate / df[col]

        df["USD_TRY"] = try_rate
        df = df.rename(columns={"USD_TRY": "TRY_per_USD"})

    return df


def save_fx(df: pd.DataFrame, filename: str = "fx_rates.csv"):
    """Save FX data to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} FX records to {output_path}")
