"""Closed-vocabulary maps used by the dashboard to render policy events in
human-friendly American English.

Pure functions; no I/O; no LLM calls. Every helper is total — it never raises
and falls back to a sensible default on null/unknown input. This is the
single source of truth for direction/magnitude display labels.
"""
from __future__ import annotations

EVENT_TYPE_HUMAN: dict[str, str] = {
    "regulation": "Trade Policy",
    "sanction": "Export Restriction",
    "frost": "Frost Warning",
    "economic": "FX / Economic Shock",
    "supply": "Supply Disruption",
    "trade": "Trade Event",
    "pandemic": "Pandemic / Logistics",
    "demand": "Demand Shift",
    "other": "Other",
}

IMPACT_DIRECTION_HUMAN: dict[str, str] = {
    "up": "Upward",
    "down": "Downward",
    "neutral": "Neutral",
}

IMPACT_MAGNITUDE_HUMAN: dict[int | str, str] = {
    1: "Minor",
    2: "Moderate",
    3: "Major",
    "1": "Minor",
    "2": "Moderate",
    "3": "Major",
}


def event_type_to_human(raw: str | None) -> str:
    """Return the display string for a legacy event_type token."""
    if not raw:
        return "Other"
    return EVENT_TYPE_HUMAN.get(str(raw).lower(), "Other")


def direction_to_human(raw: str | None) -> str:
    """Return the display string for an impact_direction value."""
    if not raw:
        return "Neutral"
    return IMPACT_DIRECTION_HUMAN.get(str(raw).lower(), "Neutral")


def magnitude_to_human(raw: int | str | None) -> str:
    """Return the display string for an impact_magnitude value (1/2/3 → Minor/Moderate/Major)."""
    if raw is None:
        return "Minor"
    return IMPACT_MAGNITUDE_HUMAN.get(raw, IMPACT_MAGNITUDE_HUMAN.get(str(raw), "Minor"))
