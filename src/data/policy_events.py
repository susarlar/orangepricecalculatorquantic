"""
Turkish agricultural policy and news event data.

Encodes known policy events, trade regulations, and market shocks
that historically impacted orange prices. These act as binary/categorical
features for the prediction model.

Sources:
- Turkish Ministry of Agriculture (TAGEM) announcements
- Official Gazette (Resmi Gazete) trade regulations
- USDA FAS GAIN reports on Turkey citrus
- News archives on frost events, export bans, tariff changes
"""
import logging
from datetime import datetime

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


# ─── Known Policy & Market Events ────────────────────────────────────────────────

POLICY_EVENTS = [
    # Format: (date, event_type, description, impact_direction, magnitude)
    # impact_direction: "up" = price increase, "down" = price decrease
    # magnitude: 1-3 (1=minor, 2=moderate, 3=major)

    # ── Export/Import regulations ──
    ("2008-01-01", "regulation", "Rusya'ya narenciye ihracat protokolü yenilendi", "up", 1),
    ("2009-07-01", "regulation", "AB ile gümrük birliği tarım ürünleri genişletildi", "down", 1),
    ("2014-08-01", "sanction", "Rusya AB/ABD gıda ambargosu — Türk narenciyeye talep artışı", "up", 3),
    ("2015-11-24", "sanction", "Rusya-Türkiye krizi — narenciye ihracatı durduruldu", "down", 3),
    ("2016-06-29", "sanction", "Rusya-Türkiye ilişkileri normalleşme başladı", "up", 2),
    ("2017-01-01", "regulation", "Rusya'ya narenciye ihracatı tam açıldı", "up", 2),
    ("2020-03-15", "pandemic", "COVID-19 — lojistik aksama, hal kapanmaları", "up", 2),
    ("2020-06-01", "pandemic", "COVID sonrası talep artışı — sağlıklı beslenme trendi", "up", 1),
    ("2022-02-24", "sanction", "Rusya-Ukrayna savaşı — Türk narenciyeye yönelim", "up", 2),
    ("2022-06-01", "regulation", "Rusya liman kısıtlamaları — alternatif rota maliyeti", "up", 1),
    ("2023-01-01", "regulation", "Irak gümrük vergisi artışı — ihracat maliyeti yükseldi", "down", 2),
    ("2024-01-01", "regulation", "AB pestisit kalıntı limitleri sıkılaştırıldı", "down", 1),
    ("2024-09-01", "regulation", "Mısır narenciye ihracat sübvansiyonu artırdı", "down", 2),

    # ── Frost/climate events ──
    ("2008-01-15", "frost", "Antalya bölgesi ağır don olayı", "up", 3),
    ("2012-02-01", "frost", "Akdeniz bölgesi soğuk dalgası", "up", 2),
    ("2016-01-10", "frost", "Finike bölgesi don hasarı", "up", 2),
    ("2019-01-08", "frost", "Batı Akdeniz don olayı — rekolte kaybı %15", "up", 2),
    ("2021-02-15", "frost", "Antalya kar yağışı — narenciye hasarı", "up", 2),
    ("2024-01-20", "frost", "Mersin-Adana don olayı — erken hasat zorlandı", "up", 2),

    # ── Economic/monetary events ──
    ("2018-08-01", "economic", "TL krizi — döviz fiyat artışı, girdi maliyeti yükseldi", "up", 3),
    ("2019-07-01", "economic", "MB faiz indirimi — TL değer kaybı devam", "up", 1),
    ("2021-12-01", "economic", "TL sert değer kaybı — girdi+nakliye maliyeti patlaması", "up", 3),
    ("2022-06-01", "economic", "Mazot ve gübre fiyatları %100+ artış", "up", 3),
    ("2023-06-01", "economic", "MB ortodoks politikaya dönüş — TL stabilize", "down", 1),
    ("2023-07-01", "economic", "Asgari ücret artışı — işçilik maliyeti yükseldi", "up", 1),
    ("2024-01-01", "economic", "Asgari ücret %49 artış — hasat işçiliği maliyeti", "up", 2),
    ("2025-01-01", "economic", "Asgari ücret %30 artış", "up", 1),

    # ── Supply/production events ──
    ("2020-05-01", "supply", "COVID — mevsimlik işçi sıkıntısı, hasat gecikmesi", "up", 2),
    ("2021-07-01", "supply", "Antalya orman yangınları — bazı bahçeler etkilendi", "up", 1),
    ("2023-02-06", "supply", "Kahramanmaraş depremi — Hatay narenciye bölgesi etkilendi", "up", 2),
    ("2023-09-01", "supply", "Akdeniz'de kuraklık — su kısıtlamaları", "up", 2),
    ("2024-04-01", "supply", "Finike bölgesi dolu hasarı", "up", 1),

    # ── Trade/market developments ──
    ("2019-10-01", "trade", "Türkiye Suriye operasyonu — bazı pazarlarla ilişki gerginliği", "down", 1),
    ("2021-03-01", "trade", "Süveyş Kanalı tıkanması — deniz taşımacılığı aksadı", "up", 1),
    ("2023-10-07", "trade", "İsrail-Gazze çatışması — rakip ihracatçı devre dışı", "up", 2),
    ("2024-12-01", "trade", "Kızıldeniz krizi — navlun maliyeti artışı", "up", 1),
]


def build_policy_events_df() -> pd.DataFrame:
    """Convert policy events list to a DataFrame.

    Returns:
        DataFrame with date, event_type, description, impact_direction, magnitude.
    """
    rows = []
    for date_str, event_type, desc, direction, magnitude in POLICY_EVENTS:
        rows.append({
            "date": pd.Timestamp(date_str),
            "event_type": event_type,
            "description": desc,
            "impact_direction": direction,
            "impact_magnitude": magnitude,
            "impact_sign": 1 if direction == "up" else -1,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def build_policy_features(start_date: str = "2007-01-01", end_date: str = None) -> pd.DataFrame:
    """Build daily policy/event features for merging with price data.

    Creates binary and cumulative features from known events:
    - event_active: whether any event is active in the recent window
    - cumulative impact score in rolling windows
    - event type dummies

    Args:
        start_date: Feature start date.
        end_date: Feature end date (default: today).

    Returns:
        Daily DataFrame with policy features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    events = build_policy_events_df()

    # Create daily date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    daily = pd.DataFrame({"date": dates})

    # For each event, create an impact window (event effects linger)
    # Major events: 90 days, moderate: 60 days, minor: 30 days
    impact_windows = {1: 30, 2: 60, 3: 90}

    daily["policy_impact_score"] = 0.0
    daily["policy_event_count_30d"] = 0
    daily["policy_event_count_90d"] = 0

    # Event type columns
    for etype in ["regulation", "sanction", "frost", "economic", "supply", "trade", "pandemic"]:
        daily[f"event_{etype}_active"] = 0

    for _, event in events.iterrows():
        event_date = event["date"]
        window_days = impact_windows.get(event["impact_magnitude"], 30)
        sign = event["impact_sign"]
        magnitude = event["impact_magnitude"]

        # Create decaying impact over the window
        for day_offset in range(window_days):
            target_date = event_date + pd.Timedelta(days=day_offset)
            mask = daily["date"] == target_date
            if mask.any():
                # Exponential decay: full impact at day 0, ~37% at window_days/2
                decay = magnitude * sign * (0.5 ** (day_offset / (window_days / 2)))
                daily.loc[mask, "policy_impact_score"] += decay
                daily.loc[mask, f"event_{event['event_type']}_active"] = 1

    # Rolling event counts — mark event dates
    event_marker = pd.Series(0, index=daily.index)
    for _, event in events.iterrows():
        mask = daily["date"] == event["date"]
        event_marker[mask] += 1

    # Smooth the event counts with rolling sums
    daily["policy_event_count_30d"] = event_marker.rolling(30, min_periods=1).sum()
    daily["policy_event_count_90d"] = event_marker.rolling(90, min_periods=1).sum()

    # Smooth impact score
    daily["policy_impact_30d_avg"] = daily["policy_impact_score"].rolling(30, min_periods=1).mean()
    daily["policy_impact_90d_avg"] = daily["policy_impact_score"].rolling(90, min_periods=1).mean()

    return daily


def save_policy_events(df: pd.DataFrame, filename: str = "policy_events.csv"):
    """Save policy events to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} policy event records to {output_path}")


def save_policy_features(df: pd.DataFrame, filename: str = "policy_features.csv"):
    """Save daily policy features to CSV."""
    output_path = RAW_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} policy feature records to {output_path}")
