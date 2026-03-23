"""
Hal (wholesale market) price data collection for Turkish orange markets.

Sources:
1. IBB tarim.ibb.istanbul — Istanbul market API (historical back to 2004)
   Uses GUID-based product IDs and pipe-delimited response format.
2. CollectAPI — Simple JSON API for current prices (requires API key)
"""
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

# ─── IBB Istanbul Hal API ───────────────────────────────────────────────────────

IBB_BASE_URL = "https://tarim.ibb.istanbul/inc/halfiyatlari"
IBB_AUTH = {
    "tUsr": "M3yV353bZe",
    "tPas": "LA74sBcXERpdBaz",
    "tVal": "881f3dc3-7d08-40db-b45a-1275c0245685",
    "HalTurId": "2",  # Meyve/Sebze hali
}
IBB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://tarim.ibb.istanbul/avrupa-yakasi-hal-mudurlugu/hal-fiyatlari.html",
}

# Product GUIDs from the IBB hal fiyatlari dropdown
PORTAKAL_PRODUCTS = {
    "portakal": "6bb77263-19e5-400f-9b3c-fc59886c3b57",           # Generic (2007-2026, best coverage)
    "portakal_washington": "fd97bf00-8bcf-495b-a121-d880f435b81f", # Washington Navel
    "portakal_finike": "3635a278-d859-40cd-8b3d-0acc47fff76b",    # Finike Çavdır
    "portakal_valensiya": "4c8f2f01-7ccb-47f7-a57e-eafd3162f1c0", # Valencia
    "portakal_sikmalik": "ebbbe553-8461-4085-bd3d-ee04c4bdb68d",  # Sıkmalık (juice)
    "portakal_lux": "f1f41075-146f-4149-8015-1ce89667cc5e",       # Lüx grade
    "portakal_2kalite": "f672d9e1-97df-4f18-b9a2-e03853e6fa3c",   # 2nd quality
    "portakal_alt": "b1aa9404-3f52-4c49-bd8c-c6f8281e5e87",       # Alternate entry
}

# Daily category IDs
KATEGORI_IDS = {
    "meyve": "5",
    "sebze": "6",
    "ithal": "7",
}


def fetch_ibb_daily(date: str, kategori: str = "meyve") -> pd.DataFrame:
    """Fetch daily prices from IBB Istanbul hal.

    Args:
        date: Date string in dd.mm.yyyy format.
        kategori: 'meyve', 'sebze', or 'ithal'.

    Returns:
        DataFrame with columns: product, unit, min_price, max_price, date.
    """
    url = f"{IBB_BASE_URL}/gunluk_fiyatlar.asp"
    kategori_id = KATEGORI_IDS.get(kategori, kategori)
    params = {**IBB_AUTH, "tarih": date, "kategori": kategori_id}

    try:
        resp = requests.get(url, params=params, headers=IBB_HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")

        if not table:
            logger.warning(f"No table found for {date}")
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr")[1:]:  # skip header
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) >= 4:
                rows.append({
                    "product": cells[0],
                    "unit": cells[1],
                    "min_price": _parse_turkish_number(cells[2]),
                    "max_price": _parse_turkish_number(cells[3]),
                    "date": datetime.strptime(date, "%d.%m.%Y").date(),
                    "market": "istanbul",
                })

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"IBB daily fetch failed for {date}: {e}")
        return pd.DataFrame()


def fetch_ibb_monthly(year: int, month: int, product_guid: str = None) -> pd.DataFrame:
    """Fetch monthly price data from IBB.

    Args:
        year: Year (e.g., 2025).
        month: Month (1-12).
        product_guid: Product GUID. Defaults to generic Portakal.

    Returns:
        DataFrame with daily prices for the month.
    """
    if product_guid is None:
        product_guid = PORTAKAL_PRODUCTS["portakal"]

    url = f"{IBB_BASE_URL}/aylik_fiyatlar.asp"
    params = {**IBB_AUTH, "yil": str(year), "ay": str(month), "urun": product_guid}

    try:
        resp = requests.get(url, params=params, headers=IBB_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or "+-+-+" not in text:
            logger.warning(f"No monthly data for {year}-{month:02d}")
            return pd.DataFrame()

        parts = text.split("+-+-+")
        if len(parts) < 3:
            return pd.DataFrame()

        dates = parts[0].split("|")
        min_prices = parts[1].split("|")
        max_prices = parts[2].split("|")

        rows = []
        for i in range(min(len(dates), len(min_prices), len(max_prices))):
            date_str = dates[i].strip()
            min_p = _parse_turkish_number(min_prices[i])
            max_p = _parse_turkish_number(max_prices[i])
            if min_p == 0 and max_p == 0:
                continue
            rows.append({
                "date": date_str,
                "min_price": min_p,
                "max_price": max_p,
                "product": "portakal",
                "market": "istanbul",
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
            df = df.dropna(subset=["date"])
        return df

    except Exception as e:
        logger.error(f"IBB monthly fetch failed for {year}-{month:02d}: {e}")
        return pd.DataFrame()


def fetch_ibb_yearly(year: int, product_guid: str = None) -> pd.DataFrame:
    """Fetch yearly aggregated prices from IBB (monthly min/max for a year).

    Args:
        year: Year (e.g., 2025).
        product_guid: Product GUID. Defaults to generic Portakal.

    Returns:
        DataFrame with monthly price aggregates.
    """
    if product_guid is None:
        product_guid = PORTAKAL_PRODUCTS["portakal"]

    url = f"{IBB_BASE_URL}/yillik_fiyatlar.asp"
    params = {**IBB_AUTH, "yil": str(year), "urun": product_guid}

    try:
        resp = requests.get(url, params=params, headers=IBB_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or "+-+-+" not in text:
            return pd.DataFrame()

        parts = text.split("+-+-+")
        if len(parts) < 3:
            return pd.DataFrame()

        months = parts[0].split("|")
        min_prices = parts[1].split("|")
        max_prices = parts[2].split("|")
        units = parts[3].split("|") if len(parts) > 3 else []

        rows = []
        for i in range(min(len(months), len(min_prices), len(max_prices))):
            min_p = _parse_turkish_number(min_prices[i])
            max_p = _parse_turkish_number(max_prices[i])
            if min_p == 0 and max_p == 0:
                continue
            rows.append({
                "month": months[i].strip(),
                "min_price": min_p,
                "max_price": max_p,
                "unit": units[i].strip() if i < len(units) else "Kilogram",
                "product": "portakal",
                "market": "istanbul",
                "year": year,
            })

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"IBB yearly fetch failed for {year}: {e}")
        return pd.DataFrame()


def fetch_ibb_all_years(product_guid: str = None) -> pd.DataFrame:
    """Fetch all historical yearly averages from IBB (2004-present).

    Args:
        product_guid: Product GUID. Defaults to generic Portakal.

    Returns:
        DataFrame with yearly min/max prices.
    """
    if product_guid is None:
        product_guid = PORTAKAL_PRODUCTS["portakal"]

    url = f"{IBB_BASE_URL}/tum_yillarin_fiyatlari.asp"
    params = {**IBB_AUTH, "urun": product_guid}

    try:
        resp = requests.get(url, params=params, headers=IBB_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or "+-+-+" not in text:
            return pd.DataFrame()

        parts = text.split("+-+-+")
        if len(parts) < 3:
            return pd.DataFrame()

        years = parts[0].split("|")
        min_prices = parts[1].split("|")
        max_prices = parts[2].split("|")
        units = parts[3].split("|") if len(parts) > 3 else []

        rows = []
        for i in range(min(len(years), len(min_prices), len(max_prices))):
            min_p = _parse_turkish_number(min_prices[i])
            max_p = _parse_turkish_number(max_prices[i])
            rows.append({
                "year": int(years[i].strip()),
                "min_price": min_p,
                "max_price": max_p,
                "unit": units[i].strip() if i < len(units) else "Kilogram",
                "product": "portakal",
                "market": "istanbul",
            })

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"IBB all years fetch failed: {e}")
        return pd.DataFrame()


def fetch_all_portakal_varieties() -> pd.DataFrame:
    """Fetch all-years data for every portakal variety and combine."""
    all_data = []
    for variety, guid in PORTAKAL_PRODUCTS.items():
        logger.info(f"Fetching all-years data for {variety}")
        df = fetch_ibb_all_years(guid)
        if not df.empty:
            df["variety"] = variety
            all_data.append(df)
        time.sleep(0.5)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# ─── CollectAPI (current prices by city) ────────────────────────────────────────

COLLECTAPI_URL = "https://api.collectapi.com/bazaar/single"


def fetch_collectapi(city: str, api_key: str) -> pd.DataFrame:
    """Fetch current hal prices from CollectAPI.

    Args:
        city: Turkish city name (e.g., 'antalya', 'mersin').
        api_key: CollectAPI key.

    Returns:
        DataFrame with current prices.
    """
    headers = {
        "Authorization": f"apikey {api_key}",
        "Content-Type": "application/json",
    }
    params = {"city": city}

    try:
        resp = requests.get(COLLECTAPI_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            logger.warning(f"CollectAPI returned no data for {city}")
            return pd.DataFrame()

        rows = []
        for item in data.get("result", []):
            rows.append({
                "product": item.get("isim", ""),
                "min_price": _parse_turkish_number(item.get("min", "0")),
                "max_price": _parse_turkish_number(item.get("max", "0")),
                "unit": item.get("birim", ""),
                "market": item.get("hal", city),
                "date": datetime.now().date(),
            })

        df = pd.DataFrame(rows)
        return df[df["product"].str.contains("portakal", case=False, na=False)]

    except Exception as e:
        logger.error(f"CollectAPI fetch failed for {city}: {e}")
        return pd.DataFrame()


# ─── Bulk collection ────────────────────────────────────────────────────────────


def collect_historical_prices(
    start_year: int = 2007,
    end_year: int = None,
    product_guid: str = None,
) -> pd.DataFrame:
    """Collect historical orange prices from IBB monthly endpoint.

    Args:
        start_year: First year to collect (data available from 2007).
        end_year: Last year to collect (default: current year).
        product_guid: Product GUID. Defaults to generic Portakal.

    Returns:
        Combined DataFrame with all historical daily price data.
    """
    if product_guid is None:
        product_guid = PORTAKAL_PRODUCTS["portakal"]
    if end_year is None:
        end_year = datetime.now().year

    all_data = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > datetime.now().month:
                break
            logger.info(f"Fetching IBB portakal {year}-{month:02d}")
            df = fetch_ibb_monthly(year, month, product_guid)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.3)  # rate limiting

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined["avg_price"] = (combined["min_price"] + combined["max_price"]) / 2
        return combined

    return pd.DataFrame()


def save_prices(df: pd.DataFrame, filename: str = "hal_prices.csv"):
    """Save price data to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} price records to {output_path}")


# ─── Utilities ──────────────────────────────────────────────────────────────────


def _parse_turkish_number(s: str) -> float:
    """Parse Turkish number format (comma decimal, dot thousands)."""
    if not s or not isinstance(s, str):
        return 0.0
    try:
        cleaned = s.strip().replace(".", "").replace(",", ".")
        return float(cleaned)
    except ValueError:
        return 0.0


def filter_oranges(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only orange products."""
    if df.empty:
        return df
    mask = df["product"].str.contains("portakal|orange", case=False, na=False)
    return df[mask].copy()
