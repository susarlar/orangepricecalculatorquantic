"""Contract tests for src.utils.descriptions."""
from src.utils.descriptions import DESCRIPTIONS


def test_market_news_feed_key_present():
    assert "market_news_feed" in DESCRIPTIONS
    assert len(DESCRIPTIONS["market_news_feed"]) > 40
