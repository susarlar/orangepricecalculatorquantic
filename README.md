# Orange Price Predictor

> **MSc Software Engineering Capstone — Quantic School of Business and Technology**
> AI-augmented decision-support system that forecasts Finike orange wholesale prices 7–90 days ahead.

[![Daily refresh](https://github.com/susarlar/orangepricecalculatorquantic/actions/workflows/daily-update.yml/badge.svg)](https://github.com/susarlar/orangepricecalculatorquantic/actions/workflows/daily-update.yml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue)](./tests)
[![Deploy: Render](https://img.shields.io/badge/deploy-Render-46e3b7)](https://render.com)

---

## Capstone Submission Links

| Item | Where |
|---|---|
| **Live deployed app** | _Add the Render URL here once the service is live._ |
| **Trello task board** | _Add the public Trello URL here._ |
| **User stories (initial backlog)** | [USER_STORIES.md](./USER_STORIES.md) |
| **Design and testing document** | [DESIGN_AND_TESTING.md](./DESIGN_AND_TESTING.md) |
| **Sprint records (≥3 sprints)** | [SPRINTS.md](./SPRINTS.md) |
| **Recorded 15–20 min demo** | _Submitted via the Quantic dashboard._ |
| **Repository access** | This repo must be shared with the GitHub user `quantic-grader`. |

> **Quantic grader access:** the repository owner adds `quantic-grader` as a collaborator under **Settings → Collaborators and teams**.

---

## What it does

A farmer in Finike, a trader in Mersin, an exporter in Antalya, or an analyst in Ankara opens a single web dashboard and gets:

- Today's Antalya Hal price and the breakeven cost.
- A SELL / WAIT / COLD STORAGE / SELL NOW recommendation with a written rationale.
- 7 / 14 / 30 / 60 / 90-day price forecasts with P10–P90 intervals.
- A live alert feed for frost, drought, FX shocks, satellite stress, and seasonal-calendar events.
- Backtested model accuracy, predicted-vs-actual tracking, and SHAP feature importance.
- Year-over-year, monthly seasonality, volatility, and FX-overlay analytics.

All of it is refreshed daily by GitHub Actions and served from a free Render dyno.

## Architecture in one paragraph

A modular monolith with a Pipeline / Pipes-and-Filters internal structure: ten Repository-pattern collectors fan in to a single feature builder, three Strategy-pattern model families train against the same feature matrix, and a composite alert runner produces a sorted alert list. Every stage persists its output to disk so any single one can be re-run independently. A Streamlit dashboard reads those artifacts; a daily GitHub Action recomputes them. Full rationale, pattern catalog, deployment trade-offs, and testing approach are in [DESIGN_AND_TESTING.md](./DESIGN_AND_TESTING.md).

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
python -m playwright install chromium    # only if running the Antalya scraper

# 2. Collect data → features → train → alerts
python -m src.pipeline --all

# 3. Run the farmer model (Antalya target)
python -c "from src.models.farmer import train_all_farmer_models; train_all_farmer_models()"

# 4. Start the dashboard
streamlit run dashboard.py
# Public read-only dashboard — no authentication required
```

## Daily operation (production-style)

```bash
# Idempotent refresh + retrain + predict + alerts. Used by GitHub Actions.
python -m src.auto_refresh --full
```

## Tests

```bash
pytest tests/ -v
```

26 tests cover config integrity, breakeven math, season-phase mapping, decision logic (SELL / WAIT / COLD STORAGE / SELL NOW), the alert dataclass and rule engine, and policy event schema validation (including a guard that ensures policy descriptions stay in English).

## Deployment

`render.yaml` is committed at the root. Connect the repo in Render and the dashboard deploys as a public read-only web service — no environment variables required.

## Repository layout

```
orangepricepredictor/
├── .github/workflows/         # daily-update + weekly-update CI
├── .ctoc/                     # CTOC plugin settings
├── data/
│   ├── raw/                   # API outputs (gitignored where >10 MB)
│   └── processed/             # feature matrix, predictions, alerts
├── docs/
│   └── data_sources.md        # English data sources + features doc
├── models/                    # Saved .joblib artifacts
├── notebooks/                 # EDA notebooks
├── plans/                     # CTOC functional/implementation/execution plans
├── src/
│   ├── alerts/                # Scenario alert rules
│   ├── data/                  # Per-source collectors (Repository pattern)
│   ├── features/              # Feature engineering
│   ├── models/                # baseline / advanced / farmer (Strategy)
│   ├── auto_refresh.py        # CI orchestrator
│   ├── pipeline.py            # CLI: collect / features / train / alerts
│   ├── prediction_tracker.py  # Live-accuracy logger
│   └── config.py              # Region, thresholds, horizons
├── tests/                     # pytest suite
├── CLAUDE.md                  # Project conventions for Claude Code
├── DESIGN_AND_TESTING.md      # Architecture + testing decisions
├── IRON_LOOP.md               # Iron Loop session state
├── README.md                  # This file
├── SPRINTS.md                 # Sprint records
├── USER_STORIES.md            # Product backlog
├── dashboard.py               # Streamlit application
├── render.yaml                # Render deployment manifest
├── requirements.txt           # Full Python dependencies
└── requirements-deploy.txt    # Slim deploy-only dependencies
```

## License and attribution

Academic project. Data is sourced from public APIs:
- İBB Istanbul Hal price archive
- Antalya Belediyesi Hal price page (scraped politely with rate limits)
- Open-Meteo (weather)
- Frankfurter (FX)
- FAO GIEWS, USDA FAS, European Commission (foreign markets)
- Sentinel-2 (NDVI; synthetic proxy used where credentials are absent)

All third-party data remains the property of its provider.
