# User Stories — Orange Price Predictor

> Initial product backlog produced by the Product Owner during Sprint 0.
> Format: INVEST stories with Gherkin (Given/When/Then) acceptance criteria.
> Prioritization: MoSCoW (Must / Should / Could / Won't).
> Each story carries a story-point estimate (Fibonacci) and a planned sprint.

---

## Personas

- **Selma — Finike farmer.** Owns 4 ha of Washington Navel and Valencia trees. Decides daily whether to harvest, sell to Antalya Hal, or place fruit in cold storage. Smartphone-only user, limited time, needs clear "act now / wait" guidance.
- **Mert — Wholesale trader (komisyoncu).** Buys at Antalya/Mersin Hal, resells to Istanbul. Wants 30–90 day price-direction signals to plan inventory and storage.
- **Defne — Exporter, Antalya.** Ships to Iraq, EU, Russia. Watches FX, importer regulations, and competitor-country supply (Egypt, Spain, South Africa).
- **Ahmet — Agricultural analyst, Ministry of Agriculture.** Tracks structural and policy effects on citrus prices for monthly briefings.
- **Quantic grader.** External evaluator who must verify functional capabilities through the dashboard and repository.

---

## Epic Map

| Epic | Goal | Personas |
|------|------|----------|
| E1. Data foundation | Reliable, fresh, multi-source feature pipeline | All |
| E2. Forecasting engine | Accurate 7–90 day price predictions with intervals | All |
| E3. Farmer decision support | Actionable sell / wait / store guidance | Selma |
| E4. Market analytics | Trend, volatility, YoY, seasonality views | Mert, Defne, Ahmet |
| E5. Risk & alerts | Frost, drought, FX, satellite, calendar alerts | Selma, Mert, Defne |
| E6. Continuous delivery | Daily refresh, auto-retrain, monitoring | All |
| E7. Operability | Auth, deploy, observability, accuracy tracking | Selma, Mert, Defne, Quantic |

---

## Sprint 1 — Foundation (Mar 23 – Apr 5, 2026)

### S1-01 — Collect İBB Istanbul Hal historical prices · MUST · 5 pts
**As** any user **I want** historical Istanbul wholesale orange prices **so that** the model has a baseline target.

```gherkin
Given the İBB Hal API endpoint and a start year (2007)
When I run `python -m src.pipeline --collect`
Then `data/raw/hal_prices.csv` contains daily min/max/avg prices
And every row has columns: date, product, market, min_price, max_price, avg_price, unit
And the data spans from 2007 to today with no future-dated rows
```

### S1-02 — Collect Finike weather (Open-Meteo) · MUST · 3 pts
**As** any user **I want** historical and forecast weather **so that** the model can capture frost and drought effects.

```gherkin
Given the Finike coordinates (36.30°N, 30.15°E)
When I run the weather collector
Then `data/raw/weather_finike.csv` contains daily temp_max/min/mean, precipitation, humidity
And a 16-day forecast horizon is appended to historical data
```

### S1-03 — Synthetic NDVI fallback · SHOULD · 3 pts
**As** a developer **I want** a deterministic NDVI proxy when Sentinel-2 is unavailable **so that** the pipeline still runs end-to-end.

```gherkin
Given Sentinel-2 credentials are missing
When I call `collect_ndvi_timeseries(use_synthetic=True)`
Then a seasonally consistent NDVI series is generated
And `data/raw/ndvi_finike.csv` is written with date and ndvi columns
```

### S1-04 — FX rates (Frankfurter) · MUST · 2 pts
**As** an exporter **I want** USD/TRY and EUR/TRY history **so that** import-price thresholds can be modeled.

```gherkin
Given Frankfurter API
When I run the FX collector
Then `data/raw/fx_rates.csv` has daily TRY_per_USD and TRY_per_EUR
And weekends are forward-filled
```

### S1-05 — Feature engineering pipeline · MUST · 5 pts
**As** the modeling team **I want** lag/rolling/seasonal features **so that** I can train time-series models.

```gherkin
Given collected price + weather + FX data
When I run `python -m src.pipeline --features`
Then `data/processed/feature_matrix.csv` exists
And it contains lag features for windows {1,7,14,30,60,90}
And it contains rolling means and stds for windows {7,14,30,60,90}
And it contains target_{7,14,21,28,30,60,90}d columns
```

### S1-06 — Baseline models (Linear, RF, XGBoost) · MUST · 5 pts
**As** the modeling team **I want** baseline models with cross-validated metrics **so that** we have a reference performance.

```gherkin
Given a feature matrix with ≥1000 rows
When I run `python -m src.pipeline --train`
Then `data/processed/model_results.csv` lists Linear, RF, XGBoost rows for each horizon
And every row reports MAE, MAPE, RMSE, R²
And cross-validation uses TimeSeriesSplit (no leakage)
```

### S1-07 — Streamlit dashboard MVP · MUST · 5 pts
**As** any user **I want** a web dashboard **so that** I can explore prices and forecasts without code.

```gherkin
Given the data and model artifacts exist
When I run `streamlit run dashboard.py`
Then a password-protected dashboard opens
And the Overview page shows latest price, min/max/spread, and a price chart
And navigation between pages works without errors
```

**Sprint 1 Goal:** End-to-end pipeline from raw data to a deployed MVP dashboard with a baseline model.

**Sprint 1 Demo:** Running `python -m src.pipeline --all` followed by `streamlit run dashboard.py` shows live prices, weather, and a baseline forecast.

---

## Sprint 2 — Modeling Depth (Apr 6 – Apr 19, 2026)

### S2-01 — Tuned LightGBM and XGBoost · MUST · 5 pts
**As** an analyst **I want** hyperparameter-tuned models **so that** forecast error is minimized.

```gherkin
Given the feature matrix
When advanced training is run
Then `lightgbm_tuned_{30,60,90}d.joblib` and `xgboost_tuned_{30,60,90}d.joblib` are saved
And tuned MAE is at least 5% better than the untuned baseline for the 30-day horizon
```

### S2-02 — Quantile regression intervals · MUST · 3 pts
**As** a farmer **I want** prediction intervals (P10/P90) **so that** I can size downside risk.

```gherkin
Given a trained point model
When quantile gradient boosters are fit
Then every forecast returns lower and upper bounds
And on a holdout, ≥75% of actuals fall within the [lower, upper] interval
```

### S2-03 — Ensemble model · SHOULD · 3 pts
**As** the modeling team **I want** an ensemble of XGB + LGB + RF **so that** prediction variance is reduced.

```gherkin
Given trained tuned models
When the ensemble is fit
Then `ensemble_{30,60,90}d.joblib` is saved
And ensemble MAE ≤ best individual model MAE on a holdout
```

### S2-04 — SHAP feature importance · SHOULD · 3 pts
**As** an analyst **I want** SHAP values for the best model **so that** I can explain forecasts.

```gherkin
Given the tuned XGBoost 30-day model
When `--advanced` is run
Then `data/processed/shap_importance.csv` lists features ranked by mean |SHAP|
And the dashboard "SHAP Analysis" tab renders the top 20
```

### S2-05 — Scenario alerts (frost, drought, FX, NDVI, calendar) · MUST · 5 pts
**As** a farmer **I want** alerts on price-moving conditions **so that** I can react before the market does.

```gherkin
Given current weather, NDVI, and FX data
When `python -m src.pipeline --alerts` is run
Then severity-sorted alerts are written to `data/processed/latest_alerts.txt`
And every alert includes title, severity, expected impact %, confidence, and lead time
```

### S2-06 — Policy event features · SHOULD · 3 pts
**As** an analyst **I want** historical policy events as features **so that** the model captures structural shocks.

```gherkin
Given the curated policy event list
When policy features are built
Then daily decay-weighted impact scores are written to `data/raw/policy_features.csv`
And event-type dummies are present for {regulation, sanction, frost, economic, supply, trade, pandemic}
```

### S2-07 — Foreign markets and competitor data · SHOULD · 3 pts
**As** an exporter **I want** EU prices, FAO indices, and competitor production **so that** competitive pressure shows up in forecasts.

```gherkin
Given USDA FAS, FAO, and EU sources
When the foreign-markets collector runs
Then `data/raw/foreign_markets.csv` contains FAO fruit index, EU orange price, competition index
```

**Sprint 2 Goal:** Production-quality modeling with explainability, intervals, and an alert system.

**Sprint 2 Demo:** Walk through Model Results, Alerts, and SHAP tabs on a deployed instance.

---

## Sprint 3 — Decision Support, CI/CD, and Polish (Apr 20 – May 3, 2026)

### S3-01 — Antalya Hal Playwright scraper · MUST · 5 pts
**As** a Finike farmer **I want** Antalya Hal prices (the closest market) **so that** advice reflects the price I actually receive.

```gherkin
Given the Antalya municipality hal page
When the Playwright scraper runs for a date range
Then `data/raw/antalya_hal_prices.csv` contains scraped citrus prices
And the scraper skips weekends and handles rate limits gracefully
```

### S3-02 — Farmer decision support model · MUST · 5 pts
**As** Selma **I want** a SELL / WAIT / COLD STORAGE / SELL NOW recommendation **so that** I know what to do today.

```gherkin
Given the latest Antalya Hal prices and forecasts
When farmer advice is generated
Then `data/processed/farmer_advice.json` contains: current_price, breakeven_price, margin, forecasts (7/14/30/60/90), and a recommendation with action, reason, urgency
And the dashboard "Farmer Panel" displays each value as a styled metric
```

### S3-03 — Cold storage scenario calculator · SHOULD · 2 pts
**As** Selma **I want** to slide a "storage days" control **so that** I can see net gain or loss for any holding period.

```gherkin
Given a current price and forecasts
When I select a storage duration on the dashboard
Then expected sale price, storage cost, and net gain are recomputed live
```

### S3-04 — Auto-refresh CI (GitHub Actions, daily) · MUST · 5 pts
**As** the team **I want** a daily CI job that pulls new data and retrains **so that** the model stays current.

```gherkin
Given the `.github/workflows/daily-update.yml` workflow
When 05:00 UTC arrives
Then the job collects new data, retrains models, commits artifacts, and pushes to main
And refresh_log.csv records each source's status
```

### S3-05 — Prediction tracker · MUST · 3 pts
**As** the team **I want** every forecast logged with its target date **so that** we can measure live accuracy.

```gherkin
Given a completed prediction run
When the tracker runs
Then `data/processed/prediction_history.csv` is appended with date_generated, horizon, target_date, predicted_price, current_price
And on the target date, actual_price and error are filled in automatically
```

### S3-06 — Deploy to Render · MUST · 2 pts
**As** the Quantic grader **I want** a public URL **so that** I can verify functionality without setup.

```gherkin
Given `render.yaml` and a Render free-tier account
When the repository is connected
Then the dashboard is reachable at a public HTTPS URL
And the URL is referenced in README.md
```

### S3-07 — Dashboard freshness banner · SHOULD · 2 pts
**As** any user **I want** an at-a-glance freshness indicator **so that** I know if data is stale.

```gherkin
Given multiple data sources with different last-update dates
When the dashboard loads
Then a banner shows today's date and any source older than 2 days
And an expandable table lists every source with its last date and a status icon
```

### S3-08 — Pytest test suite · SHOULD · 3 pts
**As** the team **I want** an automated test suite **so that** regressions are caught.

```gherkin
Given the `tests/` directory
When `pytest tests/` is run
Then unit tests pass for config constants, breakeven computation, alert dataclass, and feature-builder smoke flow
And the suite completes in under 60 seconds
```

### S3-09 — Trello task board · MUST · 1 pt
**As** the Quantic grader **I want** a public Trello board **so that** I can verify sprint planning and execution.

```gherkin
Given a public Trello board with columns Backlog / Sprint Backlog / In Progress / Review / Done
When I visit the board URL
Then every user story above has a card in the appropriate column
```

**Sprint 3 Goal:** Capstone-ready system with CI/CD, decision support, deployed dashboard, and full documentation.

**Sprint 3 Demo:** Live deployed app + GitHub Actions run history + Trello board + presentation walkthrough.

---

## Backlog (Future Sprints)

- **B-01** — Sentinel-2 real-NDVI ingestion (replace synthetic) · COULD · 5 pts
- **B-02** — Telegram / WhatsApp push alerts to farmers · COULD · 3 pts
- **B-03** — Multi-region support (Mersin, Adana, Hatay) · COULD · 5 pts
- **B-04** — Mobile-first responsive layout · COULD · 3 pts
- **B-05** — News sentiment analysis from agriculture press · COULD · 5 pts
- **B-06** — Importer-country regulation tracker (EU MRL feed, RASFF) · COULD · 3 pts
- **B-07** — A/B compare farmer recommendations against actual decisions · WON'T (this release) · 5 pts

---

## Out of Scope

- Order placement or hal commission settlement (regulated activity)
- Real-time price ticker (data sources publish daily, not intraday)
- Mobile native apps — Streamlit responsive web only

---

## Definition of Done

A story is **Done** when:

1. Code is merged to `main` and passes the quality gate (`pytest tests/`).
2. Acceptance criteria are demonstrably true in the deployed dashboard.
3. Documentation is updated (README, CLAUDE.md, or DESIGN_AND_TESTING.md as applicable).
4. The Trello card moves to **Done** with a link to the merge commit.
5. Sprint demo records the working feature.
