"""American-English display strings and Turkish-term glossary for the dashboard.

Single source of truth for label substitutions performed at the display layer.
No CSV column or upstream-pipeline names are touched here — this module only
exposes strings used in `st.metric`, `st.subheader`, Plotly `title`, trace
`name`, etc.
"""
from __future__ import annotations

import re

LABELS: dict[str, str] = {
    "freshness_istanbul_hal": "Istanbul wholesale prices",
    "freshness_antalya_hal": "Antalya wholesale prices",
    "freshness_farmer_advice": "Farmer advice (Antalya)",
    "caption_today_antalya_fresh": "Today: **{today_str}** · Antalya wholesale market latest data: {last_price_date} (fresh)",
    "info_today_antalya_slightly_stale": "Today **{today_str}**. Antalya wholesale market latest price: **{last_price_date}** ({age_days} days ago). Calculations continue with the last price.",
    "warning_today_antalya_stale": "Today **{today_str}**. Antalya wholesale market data is **{age_days} days** old (last: {last_price_date}). Check that the daily pipeline is running.",
    "metric_antalya_hal_price": "Antalya Wholesale Price (Hal)",
    "trace_antalya_actual": "Antalya wholesale (actual)",
    "chart_forecast_title": "Orange Price Forecasts — Antalya Wholesale Market",
    "cost_commission_pct": "Wholesale Market Commission (%)",
    "section_antalya_vs_istanbul_hal_prices": "Antalya vs Istanbul Wholesale Prices",
    "trace_antalya": "Antalya",
    "trace_istanbul": "Istanbul",
    "subplot_orange_hal_price": "Orange Wholesale Price (TRY/kg)",
    "sidebar_footer_ibb": "- IBB Istanbul Wholesale Market (Hal)",
}


GLOSSARY: dict[str, str] = {
    "Hal": (
        "Licensed regulated wholesale produce market in Türkiye where farmers "
        "and brokers trade fresh produce. Prices set here drive retail."
    ),
    "Narenciye": "Citrus (oranges, lemons, mandarins, grapefruits).",
    "TCMB": "Central Bank of the Republic of Türkiye — publishes the official FX rates this dashboard uses.",
    "IBB": "Istanbul Metropolitan Municipality — operates the Istanbul Hal wholesale market this dashboard tracks.",
    "Antalya/Istanbul Hal": "The two Hal wholesale markets whose prices this dashboard ingests daily.",
    "Hal commission": "Statutory commission charged by Hal brokers — a fixed-percentage line item in the farmer breakeven calculation.",
}


def render_glossary_expander(st_module) -> None:
    """Render a sidebar 'Terms' expander with one line per glossary entry.

    Accepts the Streamlit module as a parameter so this function stays unit-
    testable with a mock — no module-level Streamlit dependency.
    """
    with st_module.sidebar.expander("Terms", expanded=False):
        for term, definition in GLOSSARY.items():
            st_module.markdown(f"**{term}** — {definition}")


_TURKISH_WHOLE_WORDS = re.compile(
    r"\b(?:Hal|Narenciye|Fiyat|Bakanl|Tarım)\b",
    flags=re.IGNORECASE,
)
_ALLOWED_GLOSS = re.compile(r"\((?:Hal|Narenciye|Fiyat|Bakanl|Tarım)\)", flags=re.IGNORECASE)


def assert_no_turkish_chrome(text: str) -> bool:
    """Return True if `text` contains no bare Turkish chrome tokens.

    Whitelist: tokens that appear inside parentheses are treated as glosses
    (e.g. 'Antalya Wholesale Price (Hal)') and are permitted.
    """
    stripped = _ALLOWED_GLOSS.sub("", text)
    return _TURKISH_WHOLE_WORDS.search(stripped) is None
