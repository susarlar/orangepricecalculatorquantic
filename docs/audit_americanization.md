# Dashboard Americanization Audit

Per the functional plan `americanize-finike-dashboard-ui-string-americanization`, this
artifact maps every Turkish-chrome string in `dashboard.py` to its American-English
replacement and gates the PR with a grep sweep and a per-page manual click-through.

## Substitution table

All replacements live in `src/utils/labels.py` (the canonical source of truth shared
with sibling stubs `americanize-finike-dashboard-intro-landing-page` and
`americanize-finike-dashboard-news-events-en-translation`).

| # | dashboard.py site | Original | Replacement | LABELS key |
|---|--------|----------|-------------|------------|
| 1 | freshness item tuple | `"Istanbul Hal prices"` | `"Istanbul wholesale prices"` | `freshness_istanbul_hal` |
| 2 | freshness item tuple | `"Antalya Hal prices"` | `"Antalya wholesale prices"` | `freshness_antalya_hal` |
| 3 | `st.caption` f-string | `"... · Antalya Hal latest data: ..."` | `"... · Antalya wholesale market latest data: ..."` | `caption_today_antalya_fresh` |
| 4 | `st.info` f-string | `"... Antalya Hal latest price: ..."` | `"... Antalya wholesale market latest price: ..."` | `info_today_antalya_slightly_stale` |
| 5 | `st.warning` f-string | `"... Antalya Hal data is ..."` | `"... Antalya wholesale market data is ..."` | `warning_today_antalya_stale` |
| 6 | `st.metric` label | `"Antalya Hal Price"` | `"Antalya Wholesale Price (Hal)"` (first-mention gloss) | `metric_antalya_hal_price` |
| 7 | Plotly trace `name` | `"Antalya Hal (actual)"` | `"Antalya wholesale (actual)"` | `trace_antalya_actual` |
| 8 | Plotly chart title | `"Orange Price Forecasts — Antalya Hal"` | `"Orange Price Forecasts — Antalya Wholesale Market"` | `chart_forecast_title` |
| 9 | cost-label dict value | `"Hal Commission (%)"` | `"Wholesale Market Commission (%)"` | `cost_commission_pct` |
| 10 | `st.subheader` | `"Antalya vs Istanbul Hal Prices"` | `"Antalya vs Istanbul Wholesale Prices"` | `section_antalya_vs_istanbul_hal_prices` |
| 11 | Plotly trace `name` | `"Antalya Hal"` | `"Antalya"` | `trace_antalya` |
| 12 | Plotly trace `name` | `"Istanbul Hal"` | `"Istanbul"` | `trace_istanbul` |
| 13 | `subplot_titles` | `"Orange Hal Price (TRY/kg)"` | `"Orange Wholesale Price (TRY/kg)"` | `subplot_orange_hal_price` |
| 14 | sidebar footer markdown | `"- İBB Istanbul Hal"` | `"- IBB Istanbul Wholesale Market (Hal)"` | `sidebar_footer_ibb` |
| 15 | sidebar — glossary expander | (none) | `render_glossary_expander(st)` mounted above the page radio; renders the six-term `GLOSSARY` dict | — |

## Strings intentionally NOT touched

| Site | Token | Reason |
|------|-------|--------|
| `dashboard.py:224` | `"portakal.jpeg"` | image filename (intro page, plan A) — not display chrome |
| `dashboard.py:379, 470` | `str.contains("Portakal", case=False)` | data filter against Antalya Hal CSV product column — out of scope per AC7 (data columns unchanged) |
| `dashboard.py:1060-1061` | `trends["trend_portakal_fiyat"]` | column read from `data/raw/google_trends.csv` — out of scope per AC7 |
| `dashboard.py:28, 32` | `INTRO_COPY` mentions of "Hal" | first-mention English-paragraph context inside the intro page copy (plan A); the gloss is provided by the sentence structure, not a parenthetical |
| event `hovertemplate` (Market & Policy tab) | `ev["event_type"]`, `ev["description"]` | DeepSeek pipeline output — covered by sibling stub `americanize-finike-dashboard-news-events-en-translation` (plan C) |

## Grep audit block

```bash
# Run from project root.

# 1. Chrome tokens that must have zero unaddressed matches (only INTRO_COPY allowed):
grep -nE '\b(Hal|Narenciye|Fiyat|Bakanl|Tarım)\b' dashboard.py

# Expected output (post-substitution):
#   28:    "Washington-navel-style oranges. The fruit grown here moves through Hal "
#   32:    "by fusing daily Hal prices, Finike weather, foreign-exchange rates, "
# (Both inside INTRO_COPY — explanatory English paragraph context. Not chrome.)

# 2. Acronyms allowed in glossed form only:
grep -nE 'IBB|TCMB' dashboard.py
# Expected: zero or only sidebar_footer_ibb -- check the line is the LABELS lookup.

# 3. Data-filter survivors (must remain — out of scope):
grep -n 'Portakal\|portakal' dashboard.py
# Expected: portakal.jpeg (intro image) and str.contains("Portakal") data filters.
```

## Glossary (shared with sibling stubs)

The six canonical English glosses live in `src/utils/labels.py` as the `GLOSSARY` dict.
Sibling stubs (intro page, news translation) must import the same module rather than
duplicating definitions, mitigating the terminology-drift risk from the refined plan.

| Term | One-line English definition |
|------|------------------------------|
| Hal | Licensed regulated wholesale produce market in Türkiye where farmers and brokers trade fresh produce. Prices set here drive retail. |
| Narenciye | Citrus (oranges, lemons, mandarins, grapefruits). |
| TCMB | Central Bank of the Republic of Türkiye — publishes the official FX rates this dashboard uses. |
| IBB | Istanbul Metropolitan Municipality — operates the Istanbul Hal wholesale market this dashboard tracks. |
| Antalya/Istanbul Hal | The two Hal wholesale markets whose prices this dashboard ingests daily. |
| Hal commission | Statutory commission charged by Hal brokers — a fixed-percentage line item in the farmer breakeven calculation. |

## Manual click-through checklist (Render preview)

Open the Render preview URL. For each page in the sidebar, hover at least one data
point in every chart and check every visible label. Tick each row before merge.

| Page | Tab title | Chart titles | Axis titles | Legend entries | Hover tooltips | Metric labels |
|------|-----------|--------------|-------------|----------------|----------------|---------------|
| Welcome / About | [ ] | n/a | n/a | n/a | n/a | n/a |
| Farmer Panel | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Overview | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Price Analysis | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Weather & Environment | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Market & Policy | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Demand & Trends | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Model Results | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Forecasts & Alerts | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Note: the Market & Policy page event `hovertemplate` (Turkish event_type and description
text) is the responsibility of sibling stub `americanize-finike-dashboard-news-events-en-translation`.
Until that stub merges, English chrome here may still surround Turkish event-card text;
that is the expected intermediate state.

## Sign-off

- [ ] All 15 substitutions verified in `dashboard.py`.
- [ ] `pytest tests/test_labels.py` passes.
- [ ] Grep block above returns the expected output.
- [ ] Manual click-through checklist filled in against Render preview.
- [ ] Reviewer signature: ____________________  Date: ____________
