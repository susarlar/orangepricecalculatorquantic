"""Unit tests for the scenario alert system."""
from datetime import datetime

import pandas as pd
import pytest

from src.alerts.scenario_alerts import (
    Alert,
    AlertCategory,
    AlertSeverity,
    check_calendar_alerts,
    check_frost_alerts,
    format_alert_report,
    run_all_alerts,
)


def _make_weather(temps_min):
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(temps_min)),
        "temp_min": temps_min,
        "temp_mean": [t + 5 for t in temps_min],
        "temp_max": [t + 10 for t in temps_min],
        "precipitation": [0] * len(temps_min),
    })


def test_no_frost_alert_when_temperatures_are_warm():
    weather = _make_weather([10, 12, 8, 6, 5])
    alerts = check_frost_alerts(weather)
    assert alerts == []


def test_mild_frost_alert_fires_below_threshold():
    weather = _make_weather([2, -3, -1, 4])  # -3 trips mild frost
    alerts = check_frost_alerts(weather)
    assert len(alerts) >= 1
    titles = " ".join(a.title for a in alerts)
    assert "Frost" in titles


def test_severe_frost_marked_critical():
    weather = _make_weather([-7, -8, -6])
    alerts = check_frost_alerts(weather)
    severe = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
    assert len(severe) == 1


def test_january_calendar_alert_signals_peak_harvest():
    alerts = check_calendar_alerts(date=datetime(2026, 1, 15))
    titles = [a.title for a in alerts]
    assert any("Peak Harvest" in t for t in titles)


def test_may_calendar_alert_signals_supply_tightening():
    alerts = check_calendar_alerts(date=datetime(2026, 5, 1))
    titles = [a.title for a in alerts]
    assert any("End of Season" in t or "Supply" in t for t in titles)


def test_run_all_alerts_sorts_by_severity():
    weather = _make_weather([-7, -8])  # critical
    alerts = run_all_alerts(weather=weather, ndvi=None, fx=None,
                            date=datetime(2026, 1, 15))
    severities = [a.severity for a in alerts]
    severity_order = [AlertSeverity.CRITICAL, AlertSeverity.HIGH,
                      AlertSeverity.MEDIUM, AlertSeverity.LOW]
    indices = [severity_order.index(s) for s in severities]
    assert indices == sorted(indices), f"alerts not sorted by severity: {severities}"


def test_format_alert_report_handles_empty_list():
    report = format_alert_report([])
    assert "No active alerts" in report


def test_alert_dataclass_str_includes_severity_and_title():
    alert = Alert(
        title="Test alert",
        category=AlertCategory.WEATHER,
        severity=AlertSeverity.HIGH,
        expected_impact_pct=(10, 20),
        confidence=0.8,
        lead_time_days=(7, 14),
        description="something",
    )
    text = str(alert)
    assert "HIGH" in text
    assert "Test alert" in text
    assert "Confidence: 80%" in text
