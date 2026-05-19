"""Build the display-ready news table from the DeepSeek-classified CSV.

Pure function — no Streamlit import — so the dashboard's news section is unit-
testable. Returns None for the empty state so the caller renders an info banner
instead of an empty dataframe.
"""
from __future__ import annotations

import pandas as pd

from src.translation.event_vocab import event_type_to_human

MAX_ROWS = 25


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.map({"True": True, "False": False, True: True, False: False})
    return mapped.fillna(False).astype(bool)


def build_news_table(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a display-ready DataFrame, or None for the empty state."""
    if df is None or df.empty or "relevant" not in df.columns:
        return None
    filtered = df[_coerce_bool(df["relevant"])].copy()
    if filtered.empty:
        return None
    filtered = filtered.sort_values("date", ascending=False).head(MAX_ROWS)
    return pd.DataFrame({
        "Date": pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d"),
        "Sentiment": filtered.get("sentiment", "").astype(str).str.title(),
        "Type": filtered.get("event_type", "").map(event_type_to_human),
        "Summary": filtered.get("llm_summary", "").astype(str),
        "Source": filtered.get("link", "").astype(str),
    })
