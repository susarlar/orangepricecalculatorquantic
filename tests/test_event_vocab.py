"""Contract tests for src.translation.event_vocab — display-layer maps."""
from __future__ import annotations

from src.translation.event_vocab import (
    EVENT_TYPE_HUMAN,
    direction_to_human,
    event_type_to_human,
    magnitude_to_human,
)


def test_event_type_to_human_known_tokens():
    for raw, expected in [
        ("regulation", "Trade Policy"),
        ("sanction", "Export Restriction"),
        ("frost", "Frost Warning"),
        ("economic", "FX / Economic Shock"),
        ("supply", "Supply Disruption"),
        ("trade", "Trade Event"),
        ("pandemic", "Pandemic / Logistics"),
    ]:
        assert event_type_to_human(raw) == expected


def test_event_type_to_human_falls_back_to_other():
    assert event_type_to_human(None) == "Other"
    assert event_type_to_human("") == "Other"
    assert event_type_to_human("nonexistent-token") == "Other"


def test_event_type_human_keys_cover_all_legacy_tokens():
    legacy_tokens = {"regulation", "sanction", "frost", "economic", "supply", "trade", "pandemic"}
    assert legacy_tokens.issubset(EVENT_TYPE_HUMAN.keys())


def test_direction_to_human_known():
    assert direction_to_human("up") == "Upward"
    assert direction_to_human("down") == "Downward"
    assert direction_to_human("neutral") == "Neutral"


def test_direction_to_human_falls_back_to_neutral():
    assert direction_to_human(None) == "Neutral"
    assert direction_to_human("") == "Neutral"
    assert direction_to_human("sideways") == "Neutral"


def test_magnitude_to_human_int():
    assert magnitude_to_human(1) == "Minor"
    assert magnitude_to_human(2) == "Moderate"
    assert magnitude_to_human(3) == "Major"


def test_magnitude_to_human_str():
    assert magnitude_to_human("1") == "Minor"
    assert magnitude_to_human("2") == "Moderate"
    assert magnitude_to_human("3") == "Major"


def test_magnitude_to_human_falls_back_to_minor():
    assert magnitude_to_human(None) == "Minor"
    assert magnitude_to_human(99) == "Minor"
    assert magnitude_to_human("xyz") == "Minor"
