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
            title=f"Frost Warning — Finike at {min_temp:.1f}°C",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.HIGH if frost_days >= 3 else AlertSeverity.MEDIUM,
            expected_impact_pct=(15, 30) if frost_days >= 3 else (5, 15),
            confidence=0.85,
            lead_time_days=(14, 28),
            description=(
                f"{frost_days} frost days expected. Min temperature: {min_temp:.1f}°C. "
                f"Orange supply will fall, prices will rise."
            ),
            trigger_value=min_temp,
            threshold=ALERT_THRESHOLDS["frost_mild"],
        ))

    # Severe frost
    severe_frost = recent[recent["temp_min"] < ALERT_THRESHOLDS["frost_severe"]]
    if len(severe_frost) > 0:
        min_temp = severe_frost["temp_min"].min()

        alerts.append(Alert(
            title=f"SEVERE FROST — Finike at {min_temp:.1f}°C",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.CRITICAL,
            expected_impact_pct=(40, 80),
            confidence=0.92,
            lead_time_days=(14, 90),
            description=(
                f"Severe frost event! Major crop damage expected at {min_temp:.1f}°C. "
                f"Price impact may persist through the season."
            ),
            trigger_value=min_temp,
            threshold=ALERT_THRESHOLDS["frost_severe"],
        ))

    # Drought
    if "consecutive_dry_days" in weather.columns:
        max_dry = weather["consecutive_dry_days"].iloc[-1] if len(weather) > 0 else 0
        if max_dry >= ALERT_THRESHOLDS["drought_days"]:
            alerts.append(Alert(
                title=f"Drought Warning — {int(max_dry)} dry days",
                category=AlertCategory.WEATHER,
                severity=AlertSeverity.MEDIUM,
                expected_impact_pct=(10, 20),
                confidence=0.75,
                lead_time_days=(30, 60),
                description=(
                    f"No rainfall for {int(max_dry)} days. Water stress in trees "
                    f"may affect fruit quality and yield."
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
            title="Storm / Hail Risk",
            category=AlertCategory.WEATHER,
            severity=AlertSeverity.HIGH,
            expected_impact_pct=(15, 25),
            confidence=0.70,
            lead_time_days=(21, 42),
            description="Severe wind and rainfall. Risk of fruit damage and early drop.",
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
            title=f"Satellite Alert — NDVI down {drop_pct:.0f}%",
            category=AlertCategory.SATELLITE,
            severity=AlertSeverity.HIGH if drop_pct > 25 else AlertSeverity.MEDIUM,
            expected_impact_pct=(10, 25) if drop_pct > 25 else (5, 15),
            confidence=0.80,
            lead_time_days=(30, 90),
            description=(
                f"Vegetation health in the Finike citrus region is {drop_pct:.0f}% "
                f"below the seasonal normal. Lower yield expected."
            ),
            trigger_value=latest["ndvi_anomaly_pct"],
            threshold=-ALERT_THRESHOLDS["ndvi_drop_pct"],
        ))

    # Sustained stress
    if "ndvi_stress" in ndvi.columns:
        recent_stress = ndvi.tail(3)["ndvi_stress"].sum()
        if recent_stress >= 2:
            alerts.append(Alert(
                title="Sustained Vegetation Stress — 2+ observations",
                category=AlertCategory.SATELLITE,
                severity=AlertSeverity.HIGH,
                expected_impact_pct=(15, 30),
                confidence=0.78,
                lead_time_days=(30, 90),
                description=(
                    "Stress detected in 2 of the last 3 satellite observations. "
                    "Possible drought, disease, or frost damage."
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
                    title=f"TRY Depreciation — {change_pct:.1f}% monthly",
                    category=AlertCategory.FX,
                    severity=AlertSeverity.MEDIUM,
                    expected_impact_pct=(5, 12),
                    confidence=0.75,
                    lead_time_days=(14, 30),
                    description=(
                        f"TRY lost {change_pct:.1f}% in the last 30 days. "
                        f"Imported oranges become more expensive → domestic prices rise."
                    ),
                    trigger_value=change_pct,
                    threshold=ALERT_THRESHOLDS["fx_spike_pct"],
                ))
            else:  # TRY appreciation
                alerts.append(Alert(
                    title=f"TRY Appreciation — {abs(change_pct):.1f}% monthly",
                    category=AlertCategory.FX,
                    severity=AlertSeverity.LOW,
                    expected_impact_pct=(-8, -3),
                    confidence=0.65,
                    lead_time_days=(14, 30),
                    description=(
                        f"TRY is strengthening. Imported oranges get cheaper → "
                        f"pressure on domestic prices."
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
            title="Harvest Season Begins",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.LOW,
            expected_impact_pct=(-10, -5),
            confidence=0.85,
            lead_time_days=(0, 30),
            description="Harvest is starting. Increasing supply may push prices lower.",
        ))

    # Peak harvest
    if month in [12, 1, 2]:
        alerts.append(Alert(
            title="Peak Harvest Season",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.LOW,
            expected_impact_pct=(-15, -5),
            confidence=0.80,
            lead_time_days=(0, 14),
            description="Harvest at peak. Supply is high; prices may hit a seasonal low.",
        ))

    # Late season scarcity
    if month in [4, 5]:
        alerts.append(Alert(
            title="End of Season — Supply Tightening",
            category=AlertCategory.DEMAND,
            severity=AlertSeverity.MEDIUM,
            expected_impact_pct=(10, 25),
            confidence=0.80,
            lead_time_days=(0, 30),
            description="Harvest ending. Falling supply lifts prices.",
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
        return "No active alerts. All indicators normal."

    lines = [
        "=" * 60,
        "  ORANGE PRICE ALERT REPORT",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Total alerts: {len(alerts)}",
        "=" * 60,
        "",
    ]

    for i, alert in enumerate(alerts, 1):
        lines.append(f"--- Alert #{i} ---")
        lines.append(str(alert))
        lines.append("")

    # Net impact summary
    total_min = sum(a.expected_impact_pct[0] * a.confidence for a in alerts)
    total_max = sum(a.expected_impact_pct[1] * a.confidence for a in alerts)

    lines.extend([
        "=" * 60,
        f"  Estimated net impact: {total_min:+.1f}% to {total_max:+.1f}%",
        "=" * 60,
    ])

    return "\n".join(lines)
