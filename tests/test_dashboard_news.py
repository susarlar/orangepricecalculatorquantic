"""Smoke tests for the news-table dashboard helper."""
from __future__ import annotations

import pandas as pd

from src.dashboard_helpers.news_table import build_news_table


def _row(date, relevant=True, event_type="frost", sent="bullish"):
    return {
        "date": date,
        "title": "t",
        "link": "https://x",
        "relevant": relevant,
        "sentiment": sent,
        "event_type": event_type,
        "magnitude": 2,
        "confidence": 0.8,
        "llm_summary": "demo",
    }


def test_returns_none_when_df_missing():
    assert build_news_table(None) is None


def test_returns_none_when_empty_df():
    assert build_news_table(pd.DataFrame()) is None


def test_returns_none_when_all_irrelevant():
    df = pd.DataFrame([_row("2026-05-18", relevant=False)])
    assert build_news_table(df) is None


def test_filters_and_sorts():
    df = pd.DataFrame([
        _row("2026-05-10"),
        _row("2026-05-18"),
        _row("2026-05-15", relevant=False),
    ])
    out = build_news_table(df)
    assert list(out["Date"]) == ["2026-05-18", "2026-05-10"]
    assert out.iloc[0]["Type"] == "Frost Warning"
    assert out.iloc[0]["Sentiment"] == "Bullish"


def test_caps_at_25_rows():
    dates = pd.date_range("2026-01-01", periods=40).strftime("%Y-%m-%d")
    df = pd.DataFrame([_row(d) for d in dates])
    out = build_news_table(df)
    assert len(out) == 25


def test_csv_round_trip(tmp_path):
    p = tmp_path / "news_events.csv"
    pd.DataFrame([_row("2026-05-18")]).to_csv(p, index=False)
    df = pd.read_csv(p, parse_dates=["date"])
    out = build_news_table(df)
    assert out is not None and len(out) == 1
    assert out.iloc[0]["Type"] == "Frost Warning"
