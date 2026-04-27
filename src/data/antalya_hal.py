"""
Antalya Belediyesi Hal daily price scraper.

Source: https://www.antalya.bel.tr/tr/halden-gunluk-fiyatlar
Uses Playwright to render Vue.js dynamic content.

This is the closest public data source to Finike orange prices —
Antalya Central Hal is the main regional wholesale market.
"""
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

ANTALYA_HAL_URL = "https://www.antalya.bel.tr/tr/halden-gunluk-fiyatlar"
ANTALYA_MERKEZ_ID = "67b1db61b752f39216d8392d"

# Products to track (citrus). Keywords stay Turkish to match scraped product names.
CITRUS_KEYWORDS = ["portakal", "mandalina", "limon", "greyfurt", "narenciye"]


def scrape_antalya_daily(date_str: str, hal_id: str = ANTALYA_MERKEZ_ID) -> pd.DataFrame:
    """Scrape daily prices from Antalya Belediyesi hal page.

    Args:
        date_str: Date in dd.mm.yyyy format.
        hal_id: Hal location MongoDB ObjectID.

    Returns:
        DataFrame with product, min_price, max_price, unit, date, market.
    """
    from playwright.sync_api import sync_playwright

    url = f"{ANTALYA_HAL_URL}?halyerleri={hal_id}&fiyattarih={date_str}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2500)

            rows = page.query_selector_all("#haldengunlukfiyatlartable tbody tr")

            data = []
            for row in rows:
                cells = row.query_selector_all("td")
                cell_text = [c.inner_text().strip() for c in cells]

                if len(cell_text) >= 5:
                    product = cell_text[1]
                    min_price = _parse_price(cell_text[2])
                    max_price = _parse_price(cell_text[3])
                    unit = cell_text[4]

                    if product and min_price > 0:
                        data.append({
                            "date": datetime.strptime(date_str, "%d.%m.%Y").date(),
                            "product": product,
                            "min_price": min_price,
                            "max_price": max_price,
                            "avg_price": (min_price + max_price) / 2,
                            "unit": unit,
                            "market": "antalya",
                        })

            browser.close()

        df = pd.DataFrame(data)
        logger.info(f"Scraped {len(df)} products for {date_str} from Antalya Hal")
        return df

    except Exception as e:
        logger.error(f"Antalya Hal scrape failed for {date_str}: {e}")
        return pd.DataFrame()


def scrape_antalya_range(
    start_date: str,
    end_date: str = None,
    citrus_only: bool = True,
) -> pd.DataFrame:
    """Scrape Antalya Hal prices for a date range.

    Args:
        start_date: Start date (dd.mm.yyyy or yyyy-mm-dd).
        end_date: End date (default: today).
        citrus_only: If True, only keep citrus products.

    Returns:
        Combined DataFrame.
    """
    from playwright.sync_api import sync_playwright

    # Parse dates
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            start = datetime.strptime(start_date, fmt)
            break
        except ValueError:
            continue

    if end_date:
        for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
            try:
                end = datetime.strptime(end_date, fmt)
                break
            except ValueError:
                continue
    else:
        end = datetime.now()

    all_data = []

    # Use a single browser instance for efficiency
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            current = start
            while current <= end:
                # Skip weekends (hal is closed)
                if current.weekday() < 6:  # Mon-Sat
                    date_str = current.strftime("%d.%m.%Y")
                    url = f"{ANTALYA_HAL_URL}?halyerleri={ANTALYA_MERKEZ_ID}&fiyattarih={date_str}"

                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                        page.wait_for_timeout(2000)

                        rows = page.query_selector_all("#haldengunlukfiyatlartable tbody tr")

                        for row in rows:
                            cells = row.query_selector_all("td")
                            cell_text = [c.inner_text().strip() for c in cells]

                            if len(cell_text) >= 5:
                                product = cell_text[1]
                                min_price = _parse_price(cell_text[2])
                                max_price = _parse_price(cell_text[3])
                                unit = cell_text[4]

                                if product and min_price > 0:
                                    # Filter for citrus if requested
                                    if citrus_only and not any(
                                        kw in product.lower() for kw in CITRUS_KEYWORDS
                                    ):
                                        continue

                                    all_data.append({
                                        "date": current.date(),
                                        "product": product,
                                        "min_price": min_price,
                                        "max_price": max_price,
                                        "avg_price": (min_price + max_price) / 2,
                                        "unit": unit,
                                        "market": "antalya",
                                    })

                        logger.info(f"  {date_str}: {len(rows)} products")

                    except Exception as e:
                        logger.warning(f"  {date_str}: failed ({e})")

                    time.sleep(1)  # rate limit

                current += timedelta(days=1)

            browser.close()

    except Exception as e:
        logger.error(f"Browser session failed: {e}")

    df = pd.DataFrame(all_data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Total: {len(df)} records from {start_date} to {end_date or 'today'}")

    return df


def _parse_price(s: str) -> float:
    """Parse Turkish price format: '25,00 ₺' → 25.0"""
    if not s or not isinstance(s, str):
        return 0.0
    try:
        cleaned = s.replace("₺", "").replace(".", "").replace(",", ".").strip()
        return float(cleaned)
    except ValueError:
        return 0.0


def save_antalya_prices(df: pd.DataFrame, filename: str = "antalya_hal_prices.csv"):
    """Save Antalya hal prices to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} Antalya hal records to {output_path}")
