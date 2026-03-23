# Finike Orange Price Prediction — Implementation Plan

## Goal
Predict Finike wholesale (hal) orange prices 1-3 months ahead using ML + AI, incorporating local production, weather, satellite imagery, competitor countries, trade policy, and market signals.

---

## Phase 1: Foundation — Historical Prices + Weather + Satellite
**Objective:** Build a baseline model with the highest-signal data sources.

### 1.1 Data Collection Infrastructure
- [ ] Set up `src/data/` module with scrapers/API clients
- [ ] **Hal price scraper** — Finike, Antalya, Mersin, Adana, Istanbul daily prices from hal.gov.tr or equivalent
- [ ] **Weather data pipeline** — Historical + forecast data for Finike/Antalya region (Open-Meteo API or MGM)
  - Temperature (min/max/avg), rainfall, humidity, frost events
- [ ] **Satellite data pipeline** — Sentinel-2 NDVI for Finike orange orchards via Copernicus API
  - Define region of interest (ROI) polygon for Finike citrus area
  - Extract bi-weekly NDVI composites
- [ ] Store all raw data in `data/raw/` with consistent date indexing

### 1.2 Data Processing
- [ ] `src/features/price_features.py` — Lag features, rolling averages (7d, 30d, 90d), YoY change, seasonal decomposition
- [ ] `src/features/weather_features.py` — Frost day count, cumulative rainfall, GDD, heat stress days
- [ ] `src/features/satellite_features.py` — NDVI mean/std over ROI, NDVI anomaly vs historical avg
- [ ] Unified feature matrix builder — merge all features by date

### 1.3 Baseline Model
- [ ] `src/models/baseline.py` — Simple models first:
  - Seasonal naive (last year same period)
  - Linear regression
  - Random Forest
  - XGBoost / LightGBM
- [ ] Time-series cross-validation (expanding window, no data leakage)
- [ ] Evaluation metrics: MAE, MAPE, RMSE at 1-month, 2-month, 3-month horizons
- [ ] `notebooks/01_eda.ipynb` — Exploratory data analysis
- [ ] `notebooks/02_baseline_model.ipynb` — Model training and evaluation

### 1.4 Deliverable
- Working pipeline: raw data → features → prediction
- Baseline accuracy metrics for 1/2/3 month forecasts
- EDA report with key findings

---

## Phase 2: Market Intelligence — Competitors, Trade, FX
**Objective:** Add supply/demand signals from the broader market.

### 2.1 Competitor Data
- [ ] `src/data/competitor_prices.py` — Mersin, Adana hal prices
- [ ] `src/data/international.py` — USDA FAS data for Egypt, Morocco, Spain, South Africa production/exports
- [ ] `src/data/imports_exports.py` — TÜİK foreign trade data (Turkey orange imports/exports by country)
- [ ] `src/data/fx_rates.py` — USD/TRY, EUR/TRY, EGP/TRY daily rates (TCMB or free API)

### 2.2 Trade Policy Features
- [ ] `src/features/trade_features.py`:
  - Import/export volume trends
  - Competitor country harvest overlap indicator
  - FX rate momentum and volatility
  - Seasonal tariff calendar (EU entry price system)
  - Binary flags: import ban periods, embargo events

### 2.3 Enhanced Model
- [ ] Add Phase 2 features to feature matrix
- [ ] Feature importance analysis — which new features help?
- [ ] Retrain and compare vs Phase 1 baseline
- [ ] `notebooks/03_market_features.ipynb`

### 2.4 Deliverable
- Improved model with market intelligence features
- Feature importance ranking
- Accuracy improvement report vs baseline

---

## Phase 3: AI-Powered Signals — News, Sentiment, Demand
**Objective:** Add unstructured data signals using NLP/AI.

### 3.1 News & Sentiment Pipeline
- [ ] `src/data/news_scraper.py` — Turkish agricultural news sources
- [ ] `src/features/sentiment.py` — Use LLM (DeepSeek or similar) for:
  - Classify news as positive/negative/neutral for orange prices
  - Extract key events: frost warnings, policy changes, disease outbreaks
  - Google Trends data for "portakal fiyat" and related queries
- [ ] `src/data/regulation_tracker.py`:
  - EU RASFF notifications for Turkish citrus
  - Russian import ban/quota announcements
  - Iraq trade policy changes

### 3.2 Demand Side Features
- [ ] `src/features/demand_features.py`:
  - Retail chain price tracking (if available)
  - Input cost index (fuel, fertilizer, labor composite)
  - CPI/PPI agricultural indices
  - Ramadan/holiday calendar
  - Tourism season indicator for Antalya

### 3.3 Final Model
- [ ] Ensemble model combining all phases
- [ ] Consider LSTM/Transformer for temporal patterns
- [ ] Prediction intervals (not just point forecasts)
- [ ] `notebooks/04_full_model.ipynb`
- [ ] Model explainability — SHAP values for each prediction

### 3.4 Deliverable
- Production-ready prediction pipeline
- 1-month, 2-month, 3-month forecasts with confidence intervals
- Dashboard or report generator

---

## Phase 4: Productionization (Future)
- [ ] Automated daily data refresh pipeline
- [ ] Web dashboard (Streamlit or FastAPI + React)
- [ ] Alert system for unusual price movement predictions
- [ ] Model retraining schedule
- [ ] Monitoring for model drift

---

## Technical Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Data Sources │────▶│  Processing  │────▶│   Features  │
│             │     │              │     │             │
│ - Hal API   │     │ - Clean      │     │ - Price lag │
│ - Weather   │     │ - Validate   │     │ - Weather   │
│ - Satellite │     │ - Transform  │     │ - NDVI      │
│ - Trade     │     │              │     │ - Trade     │
│ - News      │     │              │     │ - Sentiment │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │    Model    │
                                         │             │
                                         │ - XGBoost   │
                                         │ - LSTM      │
                                         │ - Ensemble  │
                                         └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │   Output    │
                                         │             │
                                         │ - 1mo pred  │
                                         │ - 2mo pred  │
                                         │ - 3mo pred  │
                                         │ - Confidence│
                                         └─────────────┘
```

## Key Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Hal price data not available via API | No target variable | Manual collection or screen scraping |
| Satellite data gaps (cloud cover) | Missing NDVI | Use compositing and interpolation |
| Limited historical data (<5 years) | Weak model | Augment with regional data, use simpler models |
| Sudden policy changes (embargos) | Unpredictable shocks | News sentiment as early warning, regime-change detection |
| Data quality issues | Garbage in, garbage out | Automated validation checks in pipeline |
