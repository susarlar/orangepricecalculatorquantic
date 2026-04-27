"""
News ingestion + DeepSeek LLM classification → daily sentiment features.

Pipeline:
1. Fetch Turkish agriculture news from Google News RSS for orange-related queries.
2. Send each article (title + summary) to DeepSeek's chat-completions endpoint
   with a structured-JSON prompt to extract: relevance, sentiment, event_type,
   magnitude, English summary, confidence.
3. Persist per-article rows to data/raw/news_events.csv.
4. Aggregate to daily features (sentiment score, volume, per-category counts,
   severity-weighted impact score) in data/raw/news_features.csv.
5. The feature builder merges these into the master feature matrix; the
   tree-based models pick them up like any other numeric column.

Graceful degradation:
- If DEEPSEEK_API_KEY is unset → logs a warning and returns empty results.
  The pipeline continues. The merged feature matrix simply has no news columns.
- If the LLM returns invalid JSON → that article is skipped, others continue.
- If Google News RSS is unreachable → returns empty; no exception bubbles up.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────────

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=tr&gl=TR&ceid=TR:tr"

# Turkish search queries — kept Turkish to match the source language.
# Quoted phrases force exact-match in Google News.
NEWS_QUERIES: list[str] = [
    "portakal fiyat",
    "narenciye ihracat",
    "Finike portakal",
    "don tarım Akdeniz",
    "Antalya hal fiyat",
    "Mersin narenciye",
    "tarım ihracat yasağı",
]

EVENT_TYPES = {"frost", "drought", "supply", "demand",
               "trade", "policy", "economic", "other"}
SENTIMENTS = {"bullish", "bearish", "neutral"}


@dataclass
class NewsArticle:
    """One scraped article before LLM classification."""
    published: pd.Timestamp
    title: str
    summary: str
    link: str
    source: str = ""


@dataclass
class ClassifiedNews:
    """A classified article ready to be written to news_events.csv."""
    date: pd.Timestamp
    title: str
    link: str
    relevant: bool
    sentiment: str            # "bullish" | "bearish" | "neutral"
    event_type: str           # one of EVENT_TYPES
    magnitude: int            # 1..3
    confidence: float         # 0..1
    llm_summary: str
    raw_summary: str = field(default="", repr=False)


# ─── Step 1: Fetch news ─────────────────────────────────────────────────────────

def fetch_news_articles(
    queries: Iterable[str] = NEWS_QUERIES,
    days_back: int = 1,
    max_per_query: int = 10,
) -> list[NewsArticle]:
    """Fetch recent articles from Google News RSS for each query.

    Args:
        queries: Search terms (Turkish).
        days_back: Skip articles older than this many days.
        max_per_query: Cap per query to control LLM cost.

    Returns:
        List of NewsArticle, deduplicated by link.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed — skipping news fetch")
        return []

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
    seen: set[str] = set()
    out: list[NewsArticle] = []

    for q in queries:
        url = GOOGLE_NEWS_RSS.format(q=q.replace(" ", "+"))
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning(f"news fetch failed for {q!r}: {e}")
            continue

        for entry in feed.entries[:max_per_query]:
            link = entry.get("link", "")
            if not link or link in seen:
                continue
            seen.add(link)

            published = _parse_published(entry)
            if published is None or published < cutoff:
                continue

            out.append(NewsArticle(
                published=published,
                title=entry.get("title", "")[:300],
                summary=_strip_html(entry.get("summary", ""))[:800],
                link=link,
                source=entry.get("source", {}).get("title", "")
                if isinstance(entry.get("source"), dict)
                else str(entry.get("source", "")),
            ))

        time.sleep(0.5)  # be polite to Google News

    logger.info(f"fetched {len(out)} unique articles across {len(list(queries))} queries")
    return out


# ─── Step 2: LLM classification ─────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = (
    "You are an agricultural commodities analyst. You read Turkish "
    "agriculture news and extract structured signals about Turkish "
    "orange (portakal) prices. You always respond with valid JSON only — "
    "no commentary, no markdown fences."
)

CLASSIFY_USER_TEMPLATE = """Analyze this article and respond with a single JSON object.

Article:
- Title: {title}
- Summary: {summary}
- Source: {source}
- Published: {published}

Respond with exactly this schema, with no extra keys and no prose:
{{
  "relevant": <true|false>,
  "sentiment": "bullish" | "bearish" | "neutral",
  "event_type": "frost" | "drought" | "supply" | "demand" | "trade" | "policy" | "economic" | "other",
  "magnitude": 1 | 2 | 3,
  "summary": "<one-sentence English summary, max 140 chars>",
  "confidence": <float 0.0 to 1.0>
}}

Definitions:
- relevant: true only if the article meaningfully bears on Turkish orange supply, demand, prices, exports, weather impacting citrus, or related policy.
- sentiment: bullish means the article points to higher orange prices in the next 1-3 months; bearish means lower; neutral means unclear or both directions.
- magnitude: 1 minor, 2 moderate, 3 major.
"""


def classify_with_deepseek(
    article: NewsArticle,
    api_key: str | None = None,
    timeout_s: float = 20.0,
) -> ClassifiedNews | None:
    """Send one article to DeepSeek's chat-completions endpoint and parse the JSON.

    Args:
        article: The scraped article.
        api_key: DeepSeek API key. Defaults to env var DEEPSEEK_API_KEY.
        timeout_s: HTTP timeout in seconds.

    Returns:
        ClassifiedNews on success, None if the LLM call fails or returns
        invalid JSON.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — skipping LLM classification")
        return None

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout_s)
    user_prompt = CLASSIFY_USER_TEMPLATE.format(
        title=article.title,
        summary=article.summary or "(no summary provided)",
        source=article.source or "unknown",
        published=article.published.strftime("%Y-%m-%d"),
    )

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as e:
        logger.warning(f"DeepSeek call failed for {article.link}: {e}")
        return None

    raw = resp.choices[0].message.content if resp.choices else ""
    return parse_classification(article, raw)


def parse_classification(article: NewsArticle, raw_json: str) -> ClassifiedNews | None:
    """Parse and validate a JSON string returned by the LLM.

    Returns None if the JSON is malformed or fails schema checks. Pure
    function — easily unit-testable without hitting any network.
    """
    if not raw_json:
        return None

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Try to recover from a common failure mode: ```json ... ``` fences
        stripped = raw_json.strip().strip("`").lstrip("json").strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(f"could not parse JSON: {raw_json[:120]!r}")
            return None

    try:
        sentiment = str(data.get("sentiment", "")).lower()
        event_type = str(data.get("event_type", "")).lower()
        magnitude = int(data.get("magnitude", 1))
        confidence = float(data.get("confidence", 0.0))
        relevant = bool(data.get("relevant", False))
        llm_summary = str(data.get("summary", ""))[:200]
    except (TypeError, ValueError):
        return None

    if sentiment not in SENTIMENTS:
        sentiment = "neutral"
    if event_type not in EVENT_TYPES:
        event_type = "other"
    magnitude = max(1, min(3, magnitude))
    confidence = max(0.0, min(1.0, confidence))

    return ClassifiedNews(
        date=article.published.normalize(),
        title=article.title,
        link=article.link,
        relevant=relevant,
        sentiment=sentiment,
        event_type=event_type,
        magnitude=magnitude,
        confidence=confidence,
        llm_summary=llm_summary,
        raw_summary=article.summary,
    )


# ─── Step 3: Build per-article DataFrame ────────────────────────────────────────

def build_news_events_df(
    articles: Iterable[NewsArticle],
    api_key: str | None = None,
) -> pd.DataFrame:
    """Classify every article and return a per-article DataFrame.

    Includes irrelevant articles too (relevant=False) so that volume metrics
    can distinguish "lots of unrelated noise" from "lots of relevant signal".
    """
    rows = []
    for art in articles:
        classified = classify_with_deepseek(art, api_key=api_key)
        if classified is not None:
            rows.append({
                "date": classified.date,
                "title": classified.title,
                "link": classified.link,
                "relevant": classified.relevant,
                "sentiment": classified.sentiment,
                "event_type": classified.event_type,
                "magnitude": classified.magnitude,
                "confidence": classified.confidence,
                "llm_summary": classified.llm_summary,
            })
        time.sleep(0.3)  # gentle rate-limit on the LLM endpoint

    if not rows:
        return pd.DataFrame(columns=[
            "date", "title", "link", "relevant", "sentiment",
            "event_type", "magnitude", "confidence", "llm_summary",
        ])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ─── Step 4: Aggregate to daily features ────────────────────────────────────────

SENTIMENT_TO_SIGN = {"bullish": 1, "bearish": -1, "neutral": 0}


def build_news_features(events: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate per-article rows into one row per day.

    The daily features:
      - news_volume: how many relevant articles fired today
      - news_sentiment_score: confidence-weighted bullish/bearish balance, in [-1, 1]
      - news_severity_score: signed magnitude, weighted by confidence
      - news_event_<type>_count: count per event_type, useful for the model

    Args:
        events: per-article DataFrame from build_news_events_df.
        today: optional pin (default: today UTC). Used to materialize a row
            even on a quiet day so the feature is never NaN-dropped.

    Returns:
        DataFrame keyed on date, ready to merge into the feature matrix.
    """
    today = today or pd.Timestamp.utcnow().normalize()

    if events is None or events.empty:
        return pd.DataFrame([{
            "date": today,
            "news_volume": 0,
            "news_sentiment_score": 0.0,
            "news_severity_score": 0.0,
            **{f"news_event_{t}_count": 0 for t in EVENT_TYPES},
        }])

    df = events.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sign"] = df["sentiment"].map(SENTIMENT_TO_SIGN).fillna(0)
    df["weighted_score"] = df["sign"] * df["confidence"]
    df["weighted_severity"] = df["sign"] * df["magnitude"] * df["confidence"]

    relevant = df[df["relevant"]]

    # Per-day aggregates
    grp = relevant.groupby("date")
    out = pd.DataFrame({
        "news_volume": grp.size(),
        "news_sentiment_score": grp["weighted_score"].mean(),
        "news_severity_score": grp["weighted_severity"].sum(),
    }).reset_index()

    # Per-event-type counts
    for etype in EVENT_TYPES:
        per_type = (
            relevant[relevant["event_type"] == etype]
            .groupby("date").size().rename(f"news_event_{etype}_count")
        )
        out = out.merge(per_type, on="date", how="left")
        out[f"news_event_{etype}_count"] = out[f"news_event_{etype}_count"].fillna(0).astype(int)

    out["news_volume"] = out["news_volume"].fillna(0).astype(int)
    out["news_sentiment_score"] = out["news_sentiment_score"].fillna(0.0).clip(-1, 1)
    out["news_severity_score"] = out["news_severity_score"].fillna(0.0)

    return out.sort_values("date").reset_index(drop=True)


# ─── Step 5: Persistence ────────────────────────────────────────────────────────

def save_news_events(df: pd.DataFrame, filename: str = "news_events.csv") -> None:
    """Append per-article rows, dedup by link, write back."""
    path = RAW_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="link", keep="last")
    else:
        combined = df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(path, index=False)
    logger.info(f"news_events.csv now has {len(combined)} rows ({len(df)} new)")


def save_news_features(df: pd.DataFrame, filename: str = "news_features.csv") -> None:
    """Append daily aggregates, dedup by date, write back."""
    path = RAW_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
    else:
        combined = df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(path, index=False)
    logger.info(f"news_features.csv now has {len(combined)} rows ({len(df)} new)")


# ─── Top-level orchestration ────────────────────────────────────────────────────

def refresh_news() -> dict:
    """Run the full news pipeline. Safe to call from CI.

    Returns a status dict with: status, articles_fetched, classified, relevant.
    Status is one of: ok, no_api_key, no_articles, error.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set — skipping news classification")
        return {"status": "no_api_key", "articles_fetched": 0,
                "classified": 0, "relevant": 0}

    try:
        articles = fetch_news_articles()
    except Exception as e:
        logger.error(f"news fetch failed: {e}")
        return {"status": "error", "error": str(e),
                "articles_fetched": 0, "classified": 0, "relevant": 0}

    if not articles:
        return {"status": "no_articles", "articles_fetched": 0,
                "classified": 0, "relevant": 0}

    events = build_news_events_df(articles, api_key=api_key)
    if not events.empty:
        save_news_events(events)
        features = build_news_features(events)
        save_news_features(features)

    return {
        "status": "ok",
        "articles_fetched": len(articles),
        "classified": len(events),
        "relevant": int(events["relevant"].sum()) if not events.empty else 0,
    }


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _parse_published(entry) -> pd.Timestamp | None:
    """Best-effort parse of an RSS entry's published timestamp."""
    for key in ("published", "updated", "pubDate"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return pd.Timestamp(raw, tz="UTC").tz_convert(None)
        except Exception:
            try:
                return pd.Timestamp(raw)
            except Exception:
                pass
    return None


def _strip_html(html: str) -> str:
    """Crude HTML strip — Google News RSS summaries are short and tag-light."""
    if not html:
        return ""
    out, in_tag = [], False
    for ch in html:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)
    return "".join(out).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result = refresh_news()
    print(json.dumps(result, indent=2))
