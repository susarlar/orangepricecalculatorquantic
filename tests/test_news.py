"""Tests for the DeepSeek-driven news pipeline.

We exercise the pure-function parts (parsing, aggregation, persistence) and
mock the network-touching pieces (RSS fetch, LLM call). No live API calls.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.news import (
    EVENT_TYPES,
    NewsArticle,
    build_news_features,
    parse_classification,
    refresh_news,
)


def _article(title="Antalya'da don alarmı", days_ago=0):
    return NewsArticle(
        published=pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days_ago),
        title=title,
        summary="Akdeniz'de soğuk hava narenciye bahçelerini etkiliyor",
        link=f"https://example.com/{title}",
        source="ExampleSource",
    )


# ─── parse_classification ────────────────────────────────────────────────────────

def test_parse_classification_happy_path():
    art = _article()
    raw = (
        '{"relevant": true, "sentiment": "bullish", "event_type": "frost",'
        ' "magnitude": 3, "summary": "Severe frost expected", "confidence": 0.9}'
    )
    out = parse_classification(art, raw)
    assert out is not None
    assert out.relevant is True
    assert out.sentiment == "bullish"
    assert out.event_type == "frost"
    assert out.magnitude == 3
    assert out.confidence == pytest.approx(0.9)
    assert out.llm_summary == "Severe frost expected"


def test_parse_classification_recovers_from_markdown_fences():
    art = _article()
    raw = (
        '```json\n{"relevant": false, "sentiment": "neutral",'
        ' "event_type": "other", "magnitude": 1, "summary": "Unrelated",'
        ' "confidence": 0.2}\n```'
    )
    out = parse_classification(art, raw)
    assert out is not None
    assert out.relevant is False


def test_parse_classification_clamps_out_of_range_values():
    art = _article()
    raw = (
        '{"relevant": true, "sentiment": "bullish", "event_type": "frost",'
        ' "magnitude": 99, "summary": "x", "confidence": 5.0}'
    )
    out = parse_classification(art, raw)
    assert out is not None
    assert out.magnitude == 3  # clamped to max
    assert out.confidence == 1.0  # clamped to max


def test_parse_classification_normalizes_unknown_categories():
    art = _article()
    raw = (
        '{"relevant": true, "sentiment": "very_bullish",'
        ' "event_type": "alien_invasion", "magnitude": 2,'
        ' "summary": "x", "confidence": 0.5}'
    )
    out = parse_classification(art, raw)
    assert out is not None
    assert out.sentiment == "neutral"
    assert out.event_type == "other"


def test_parse_classification_returns_none_for_garbage():
    art = _article()
    assert parse_classification(art, "not json at all") is None
    assert parse_classification(art, "") is None


# ─── build_news_features ─────────────────────────────────────────────────────────

def test_build_news_features_empty_input_yields_zero_row():
    df = build_news_features(pd.DataFrame())
    assert len(df) == 1
    assert df.iloc[0]["news_volume"] == 0
    assert df.iloc[0]["news_sentiment_score"] == 0
    assert df.iloc[0]["news_severity_score"] == 0
    # Every event-type column must exist even when there is no data
    for etype in EVENT_TYPES:
        assert f"news_event_{etype}_count" in df.columns


def test_build_news_features_aggregates_a_bullish_frost_day():
    today = pd.Timestamp("2026-01-15")
    events = pd.DataFrame([
        {"date": today, "title": "a", "link": "u1", "relevant": True,
         "sentiment": "bullish", "event_type": "frost", "magnitude": 3,
         "confidence": 0.9, "llm_summary": "x"},
        {"date": today, "title": "b", "link": "u2", "relevant": True,
         "sentiment": "bullish", "event_type": "frost", "magnitude": 2,
         "confidence": 0.7, "llm_summary": "y"},
    ])

    df = build_news_features(events, today=today)
    row = df[df["date"] == today].iloc[0]

    assert row["news_volume"] == 2
    # Average of (1*0.9, 1*0.7) = 0.8
    assert row["news_sentiment_score"] == pytest.approx(0.8)
    # Sum of (1*3*0.9, 1*2*0.7) = 2.7 + 1.4 = 4.1
    assert row["news_severity_score"] == pytest.approx(4.1)
    assert row["news_event_frost_count"] == 2
    assert row["news_event_drought_count"] == 0


def test_build_news_features_excludes_irrelevant_articles_from_volume():
    today = pd.Timestamp("2026-02-01")
    events = pd.DataFrame([
        {"date": today, "title": "a", "link": "u1", "relevant": True,
         "sentiment": "bearish", "event_type": "supply", "magnitude": 1,
         "confidence": 0.5, "llm_summary": "x"},
        {"date": today, "title": "b", "link": "u2", "relevant": False,
         "sentiment": "neutral", "event_type": "other", "magnitude": 1,
         "confidence": 0.2, "llm_summary": "noise"},
    ])

    df = build_news_features(events, today=today)
    row = df[df["date"] == today].iloc[0]
    assert row["news_volume"] == 1  # the irrelevant one is dropped
    assert row["news_sentiment_score"] == pytest.approx(-0.5)


def test_build_news_features_handles_mixed_sentiment_day():
    today = pd.Timestamp("2026-03-01")
    events = pd.DataFrame([
        {"date": today, "title": "a", "link": "u1", "relevant": True,
         "sentiment": "bullish", "event_type": "frost", "magnitude": 2,
         "confidence": 0.8, "llm_summary": ""},
        {"date": today, "title": "b", "link": "u2", "relevant": True,
         "sentiment": "bearish", "event_type": "supply", "magnitude": 1,
         "confidence": 0.6, "llm_summary": ""},
    ])
    df = build_news_features(events, today=today)
    row = df[df["date"] == today].iloc[0]
    # mean of (+0.8, -0.6) = 0.1
    assert row["news_sentiment_score"] == pytest.approx(0.1)
    # severity sum = 1*2*0.8 + (-1)*1*0.6 = 1.0
    assert row["news_severity_score"] == pytest.approx(1.0)


# ─── refresh_news graceful degradation ───────────────────────────────────────────

def test_refresh_news_no_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = refresh_news()
    assert result["status"] == "no_api_key"
    assert result["classified"] == 0


def test_refresh_news_no_articles_with_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with patch("src.data.news.fetch_news_articles", return_value=[]):
        result = refresh_news()
    assert result["status"] == "no_articles"
    assert result["articles_fetched"] == 0


def test_refresh_news_handles_fetch_exception(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    with patch("src.data.news.fetch_news_articles",
               side_effect=RuntimeError("network down")):
        result = refresh_news()
    assert result["status"] == "error"
    assert "network down" in result["error"]
