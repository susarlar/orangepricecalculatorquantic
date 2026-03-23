"""
Scenario-based alert system for orange price prediction.

Monitors data streams for known price-moving events and generates
alerts with expected impact, confidence, and lead time.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from src.config import ALERT_THRESHOLDS, COMPETITOR_COUNTRIES

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(Enum):
    WEATHER = "weather"
    TRADE = "trade"
    COMPETITOR = "competitor"
    DEMAND = "demand"
    FX = "fx"
    SATELLITE = "satellite"


@dataclass
class Alert:
    """A price impact alert."""
    title: str
    category: AlertCategory
    severity: AlertSeverity
    expected_impact_pct: tuple[float, float]  # (min%, max%)
    confidence: float  # 0-1
    lead_time_days: tuple[int, int]  # (min_days, max_days)
    description: str
    trigger_value: float = None
    threshold: float = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        direction = "+" if self.expected_impact_pct[0] > 0 else ""
        return (
            f"[{self.severity.value.upper()}] {self.title}\n"
            f"  Impact: {direction}{self.expected_impact_pct[0]}% to "
            f"{direction}{self.expected_impact_pct[1]}%\n"
            f"  Confidence: {self.confidence:.0%}\n"
            f"  Lead time: {self.lead_time_days[0]}-{self.lead_time_days[1]} days\n"
            f"  {self.description}"
        )


# ─── Weather alerts ─────────────────────────────────────────────────────────────

def check_frost_alerts(weather: pd.DataFrame, days_ahead: int = 16) -> list[Alert]:
    """Check for frost events in recent/forecast weather data."""
    alerts = []
    recent = weather.tail(days_ahead)

    if recent.empty:
        return alerts

    # Mild frost
    mild_frost = recent[recent["temp_min"] < ALERT_THRESHOLDS["frost_mild"]]
    if len(mild_frost) > 0:
        min_temp = mild_frost["temp_min"].min()
        frost_days = len(mild_frost)

        alerts.append(Alert(
            title=f"Don Uyarısı — Finike'de {min_temp:.1f}°C",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.HIGH if frost_days >= 3 else AlertSeverity.MEDIUM,
            expected_impact_pct=(15, 30) if frost_days >= 3 else (5, 15),
            confidence=0.85,
            lead_time_days=(14, 28),
            description=(
                f"{frost_days} gün don bekleniyor. Minimum sıcaklık: {min_temp:.1f}°C. "
                f"Portakal arzı düşecek, fiyatlar yükselecek."
            ),
            trigger_value=min_temp,
            threshold=ALERT_THRESHOLDS["frost_mild"],
        ))

    # Severe frost
    severe_frost = recent[recent["temp_min"] < ALERT_THRESHOLDS["frost_severe"]]
    if len(severe_frost) > 0:
        min_temp = severe_frost["temp_min"].min()

        alerts.append(Alert(
            title=f"ŞİDDETLİ DON — Finike'de {min_temp:.1f}°C",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.CRITICAL,
            expected_impact_pct=(40, 80),
            confidence=0.92,
            lead_time_days=(14, 90),
            description=(
                f"Şiddetli don olayı! {min_temp:.1f}°C ile ciddi ürün hasarı bekleniyor. "
                f"Sezon boyunca fiyat etkisi sürebilir."
            ),
            trigger_value=min_temp,
            threshold=ALERT_THRESHOLDS["frost_severe"],
        ))

    # Drought
    if "consecutive_dry_days" in weather.columns:
        max_dry = weather["consecutive_dry_days"].iloc[-1] if len(weather) > 0 else 0
        if max_dry >= ALERT_THRESHOLDS["drought_days"]:
            alerts.append(Alert(
                title=f"Kuraklık Uyarısı — {int(max_dry)} gün yağışsız",
                category=AlertCategory.WEATHER,
                severity=AlertSeverity.MEDIUM,
                expected_impact_pct=(10, 20),
                confidence=0.75,
                lead_time_days=(30, 60),
                description=(
                    f"Son {int(max_dry)} gündür yağış yok. Ağaçlarda su stresi "
                    f"meyve kalitesini ve verimini etkileyebilir."
                ),
                trigger_value=max_dry,
                threshold=ALERT_THRESHOLDS["drought_days"],
            ))

    # Hail (check for extreme wind + precipitation combo)
    extreme_weather = recent[
        (recent.get("wind_speed_max", pd.Series(dtype=float)) > 60) &
        (recent.get("precipitation", pd.Series(dtype=float)) > 20)
    ]
    if len(extreme_weather) > 0:
        alerts.append(Alert(
            title="Fırtına/Dolu Riski",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.HIGH,
            expected_impact_pct=(15, 25),
            confidence=0.70,
            lead_time_days=(21, 42),
            description="Şiddetli rüzgar ve yağış. Meyve hasarı ve erken dökülme riski.",
        ))

    return alerts


# ─── Satellite/NDVI alerts ──────────────────────────────────────────────────────

def check_ndvi_alerts(ndvi: pd.DataFrame) -> list[Alert]:
    """Check for vegetation stress from satellite data."""
    alerts = []

    if ndvi.empty or "ndvi_anomaly_pct" not in ndvi.columns:
        return alerts

    latest = ndvi.iloc[-1]

    # NDVI drop
    if latest.get("ndvi_anomaly_pct", 0) < -ALERT_THRESHOLDS["ndvi_drop_pct"]:
        drop_pct = abs(latest["ndvi_anomaly_pct"])
        alerts.append(Alert(
            title=f"Uydu Uyarısı — NDVI %{drop_pct:.0f} düşüş",
            category=AlertCategory.SATELLITE,
            severity=AlertSeverity.HIGH if drop_pct > 25 else AlertSeverity.MEDIUM,
            expected_impact_pct=(10, 25) if drop_pct > 25 else (5, 15),
            confidence=0.80,
            lead_time_days=(30, 90),
            description=(
                f"Finike narenciye bölgesinde bitki örtüsü sağlığı mevsim "
                f"normalinin %{drop_pct:.0f} altında. Düşük verim bekleniyor."
            ),
            trigger_value=latest["ndvi_anomaly_pct"],
            threshold=-ALERT_THRESHOLDS["ndvi_drop_pct"],
        ))

    # Sustained stress
    if "ndvi_stress" in ndvi.columns:
        recent_stress = ndvi.tail(3)["ndvi_stress"].sum()
        if recent_stress >= 2:
            alerts.append(Alert(
                title="Sürekli Bitki Stresi — 2+ gözlem",
                category=AlertCategory.SATELLITE,
                severity=AlertSeverity.HIGH,
                expected_impact_pct=(15, 30),
                confidence=0.78,
                lead_time_days=(30, 90),
                description=(
                    "Son 3 uydu gözleminin 2'sinde stres tespit edildi. "
                    "Kuraklık, hastalık veya don hasarı olabilir."
                ),
            ))

    return alerts


# ─── FX alerts ──────────────────────────────────────────────────────────────────

def check_fx_alerts(fx: pd.DataFrame) -> list[Alert]:
    """Check for significant currency movements."""
    alerts = []

    if fx.empty or "TRY_per_USD" not in fx.columns:
        return alerts

    # Monthly TRY change
    if len(fx) >= 30:
        current = fx["TRY_per_USD"].iloc[-1]
        month_ago = fx["TRY_per_USD"].iloc[-30]
        change_pct = (current - month_ago) / month_ago * 100

        if abs(change_pct) > ALERT_THRESHOLDS["fx_spike_pct"]:
            if change_pct > 0:  # TRY depreciation
                alerts.append(Alert(
                    title=f"TL Değer Kaybı — Aylık %{change_pct:.1f}",
                    category=AlertCategory.FX,
                    severity=AlertSeverity.MEDIUM,
                    expected_impact_pct=(5, 12),
                    confidence=0.75,
                    lead_time_days=(14, 30),
                    description=(
                        f"TL son 30 günde %{change_pct:.1f} değer kaybetti. "
                        f"İthal portakal pahalılaşır → yerli fiyatlar yükselir."
                    ),
                    trigger_value=change_pct,
                    threshold=ALERT_THRESHOLDS["fx_spike_pct"],
                ))
            else:  # TRY appreciation
                alerts.append(Alert(
                    title=f"TL Değer Kazancı — Aylık %{abs(change_pct):.1f}",
                    category=AlertCategory.FX,
                    severity=AlertSeverity.LOW,
                    expected_impact_pct=(-8, -3),
                    confidence=0.65,
                    lead_time_days=(14, 30),
                    description=(
                        f"TL güçleniyor. İthal portakal ucuzlar → "
                        f"yerli fiyatlarda baskı."
                    ),
                    trigger_value=change_pct,
                    threshold=-ALERT_THRESHOLDS["fx_spike_pct"],
                ))

    return alerts


# ─── Seasonal/calendar alerts ───────────────────────────────────────────────────

def check_calendar_alerts(date: datetime = None) -> list[Alert]:
    """Check for known seasonal demand events."""
    if date is None:
        date = datetime.now()

    alerts = []
    month = date.month

    # Harvest season start
    if month == 11:
        alerts.append(Alert(
            title="Hasat Sezonu Başlangıcı",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.LOW,
            expected_impact_pct=(-10, -5),
            confidence=0.85,
            lead_time_days=(0, 30),
            description="Hasat başlıyor. Artan arz ile fiyatlar düşebilir.",
        ))

    # Peak harvest
    if month in [12, 1, 2]:
        alerts.append(Alert(
            title="Hasat Sezonu Zirvesi",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.LOW,
            expected_impact_pct=(-15, -5),
            confidence=0.80,
            lead_time_days=(0, 14),
            description="Hasat zirvede. Arz yüksek, fiyatlar mevsimsel dip yapabilir.",
        ))

    # Late season scarcity
    if month in [4, 5]:
        alerts.append(Alert(
            title="Sezon Sonu — Arz Azalması",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.MEDIUM,
            expected_impact_pct=(10, 25),
            confidence=0.80,
            lead_time_days=(0, 30),
            description="Hasat sonu yaklaşıyor. Azalan arz fiyatları yükseltir.",
        ))

    return alerts


# ─── Master alert runner ────────────────────────────────────────────────────────

def run_all_alerts(
    weather: pd.DataFrame = None,
    ndvi: pd.DataFrame = None,
    fx: pd.DataFrame = None,
    date: datetime = None,
) -> list[Alert]:
    """Run all alert checks and return sorted list.

    Args:
        weather: Weather data (historical + forecast).
        ndvi: NDVI data with anomaly features.
        fx: FX rate data.
        date: Current date for calendar alerts.

    Returns:
        List of Alert objects sorted by severity.
    """
    all_alerts = []

    if weather is not None and not weather.empty:
        all_alerts.extend(check_frost_alerts(weather))

    if ndvi is not None and not ndvi.empty:
        all_alerts.extend(check_ndvi_alerts(ndvi))

    if fx is not None and not fx.empty:
        all_alerts.extend(check_fx_alerts(fx))

    all_alerts.extend(check_calendar_alerts(date))

    # Sort by severity
    severity_order = {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.HIGH: 1,
        AlertSeverity.MEDIUM: 2,
        AlertSeverity.LOW: 3,
    }
    all_alerts.sort(key=lambda a: severity_order[a.severity])

    return all_alerts


def format_alert_report(alerts: list[Alert]) -> str:
    """Format alerts into a readable report."""
    if not alerts:
        return "Aktif uyarı yok. Tüm göstergeler normal."

    lines = [
        "=" * 60,
        "  PORTAKAL FİYAT UYARI RAPORU",
        f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Toplam uyarı: {len(alerts)}",
        "=" * 60,
        "",
    ]

    for i, alert in enumerate(alerts, 1):
        lines.append(f"--- Uyarı #{i} ---")
        lines.append(str(alert))
        lines.append("")

    # Net impact summary
    total_min = sum(a.expected_impact_pct[0] * a.confidence for a in alerts)
    total_max = sum(a.expected_impact_pct[1] * a.confidence for a in alerts)

    lines.extend([
        "=" * 60,
        f"  Tahmini Net Etki: %{total_min:+.1f} ile %{total_max:+.1f} arası",
        "=" * 60,
    ])

    return "\n".join(lines)
