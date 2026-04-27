# Design and Testing Document

**Project:** Orange Price Predictor (Finike, Turkey)
**Course:** MSc Software Engineering Capstone — Quantic School of Business and Technology
**Document version:** 1.0 (Capstone submission)

This document satisfies the Quantic Capstone requirement to detail the design and architecture decisions made and the software testing implemented.

---

## 1. System Overview

The Orange Price Predictor is an **AI-augmented decision-support system** that forecasts Finike orange wholesale prices 7 to 90 days ahead and delivers actionable guidance to farmers, traders, exporters, and analysts. It integrates ten heterogeneous data sources (wholesale markets, weather, satellite, FX, policy events, foreign markets, etc.) into a single feature matrix, trains an ensemble of gradient-boosted regression models with quantile intervals, and serves the result through a password-protected Streamlit dashboard deployed on Render.

The system is operated end-to-end by a daily GitHub Actions pipeline that re-ingests data, retrains models, generates forecasts, evaluates yesterday's predictions, and commits the artifacts back to the repository.

---

## 2. Architectural Decisions

### 2.1 Style: Modular Monolith with Pipeline Architecture

We chose a **modular monolith** over microservices because:

- Single-team capstone with bounded scope; service boundaries would be premature.
- All flows are batch (daily refresh) — no need for inter-service network hops.
- Simpler observability, simpler deployment (one Render service).
- The internal module boundaries (`src/data`, `src/features`, `src/models`, `src/alerts`, `dashboard.py`) are clean enough to extract later if scale demands it.

The internal organization follows a **Pipeline / Pipes-and-Filters** pattern:

```
[ collectors ] → raw CSV → [ feature builder ] → feature matrix → [ trainer ] → joblib models
                                                                     ↓
                                                                  forecasts
                                                                     ↓
                                                  [ alerts ] + [ dashboard ] + [ tracker ]
```

Each stage reads from disk and writes to disk. This deliberate persistence is what makes every step **independently runnable, observable, and idempotent** — vital for a CI-driven daily refresh.

### 2.2 Pattern Catalog

| Pattern | Where it is used | Why |
|---|---|---|
| **Pipeline (Pipes & Filters)** | `src/pipeline.py`, `src/auto_refresh.py` | Compose stages (collect → features → train → predict → alert) with disk as the bus |
| **Repository** | `src/data/*.py` collectors | Each external source is wrapped in a single module that returns a DataFrame, hiding API/scrape mechanics |
| **Strategy** | `src/models/baseline.py`, `advanced.py`, `farmer.py` | Multiple model implementations share a uniform `train(features, horizon) → metrics` interface |
| **Adapter** | `src/data/antalya_hal.py`, `hal_prices.py` | Adapt heterogeneous APIs (REST JSON, ASP forms, JS-rendered HTML) to a common DataFrame schema |
| **Decorator (Streamlit cache)** | `dashboard.py` `@st.cache_data` | Memoize expensive `load_data()` calls per session |
| **Data Transfer Object** | `Alert` dataclass in `scenario_alerts.py` | Strongly-typed structure for severity / impact / lead-time |
| **Strategy + Composite (alerts)** | `check_frost_alerts`, `check_ndvi_alerts`, `check_fx_alerts`, `check_calendar_alerts` orchestrated by `run_all_alerts` | Each rule is a strategy; the composite runs them all and merges results |
| **Idempotent writer** | `auto_refresh.py` | Every refresh is safe to re-run; existing rows are deduplicated by date+source key |
| **Circuit-breaker (soft)** | `try/except` around each collector with status logged to `refresh_log.csv` | One failing source must not break the entire daily pipeline |
| **LLM-as-extractor (news)** | `src/data/news.py` | Google News RSS → DeepSeek `deepseek-chat` with structured-JSON response → daily aggregates merged into the feature matrix. The LLM is used as a **classifier**, not a generator — output is constrained to a fixed JSON schema and the model never sees the price target |

### 2.3 Technology Choices and Rationale

| Choice | Alternative considered | Reason |
|---|---|---|
| **Python 3.11** | Node.js, R | Best ML ecosystem; team expertise; aligns with most of the data sources' Python SDKs |
| **pandas + NumPy** | Polars, Dask | Mature, ubiquitous, sufficient for the dataset size (~10⁵ rows) |
| **scikit-learn + XGBoost + LightGBM** | PyTorch, TensorFlow | Tabular time series — gradient boosting beats deep learning consistently below ~10⁶ rows |
| **statsmodels (later)** | Prophet | More transparent decomposition, no opinionated priors |
| **SHAP** | Permutation importance | Game-theoretic, faithful to model logic, supported by XGBoost natively |
| **Streamlit** | Dash, Flask + React | Single-file deployment, fast iteration, no JS toolchain |
| **Plotly** | Matplotlib, Altair | Interactive hover, dual axes, Streamlit-native |
| **joblib** | pickle, ONNX | Best for scikit-learn / XGBoost artifacts; transparent to GitHub Actions |
| **Render free tier** | Railway, Fly.io, AWS | Free tier sufficient; Streamlit-friendly defaults; HTTPS by default |
| **GitHub Actions** | CircleCI, Jenkins | Free for public repos; tightly coupled to the source repo; no extra dashboard to learn |
| **Playwright (Antalya scraper)** | Selenium, requests + bs4 | The Antalya page renders via Vue.js; Playwright handles the JS without custom waits |
| **Open-Meteo (weather)** | Meteostat, NOAA | Free, no API key, includes 16-day forecast — critical for the alert lead-time guarantee |
| **Frankfurter (FX)** | Yahoo Finance, ECB SDMX | Free, simple, ECB-backed; sufficient for daily granularity |
| **DeepSeek `deepseek-chat`** | OpenAI GPT-4o-mini, Anthropic Claude Haiku, local Llama | OpenAI-compatible API; native JSON-mode; ~$0.14/1M input tokens — under $0.05/month at our volume; faithful Turkish→English summarisation in evaluation |
| **Google News RSS** | Manual scraping, paid news APIs | Free, stable URL contract, pre-aggregates dozens of Turkish ag-news outlets, no API key |
| **CSV persistence** | Postgres, SQLite | Diff-friendly in git; observable in the GitHub UI; no DB to host on free tier |

### 2.4 Key Design Decisions

1. **Daily batch over real-time.** All upstream sources publish daily at most. Real-time would add infrastructure cost without information gain.
2. **Disk-persisted intermediate state.** Every stage writes its output to disk so a single failure doesn't cost a full re-run, and every artifact is auditable on GitHub.
3. **Forward-fill the last Antalya price up to "today".** Without this, the farmer panel showed a blank "today" whenever the upstream Hal site lagged (which it often does on weekends). The flag `is_price_ffill` clearly marks synthesized rows so downstream code can warn.
4. **Synthetic NDVI fallback by default.** Sentinel-2 ingestion requires credentials and bandwidth that the free CI runner does not have. A deterministic seasonally-correct proxy keeps the pipeline self-contained while leaving a clear extension point (story B-01).
5. **Quantile regression for intervals (not bootstrap).** Bootstrap intervals on time-series data are biased; quantile gradient boosting is faster, leakage-free, and gives well-calibrated bounds.
6. **Log-target for tree models.** Prices are right-skewed and span 1–2 orders of magnitude across a season; `log1p` stabilizes training, `expm1` recovers price space.
7. **Time-series cross-validation, never random K-fold.** Random folds would leak future into past. We use `TimeSeriesSplit` everywhere a model is evaluated.
8. **Public read-only dashboard, no auth.** The deliverable is intended for evaluation and demonstration. Real auth (OAuth or per-user accounts) is parked in the backlog for any future operational rollout.

### 2.5 Component Diagram (text)

```
                       ┌─────────────────────────────────────┐
                       │   GitHub Actions: daily-update.yml  │
                       │   schedule: 05:00 UTC               │
                       └────────────────┬────────────────────┘
                                        │ runs
                                        ▼
                          ┌──────────────────────────┐
                          │  src/auto_refresh.py     │
                          │  (orchestrator)          │
                          └────┬─────────────────────┘
                               │
        ┌──────────┬───────────┼──────────┬──────────────┐
        ▼          ▼           ▼          ▼              ▼
  hal_prices   weather    fx_rates   antalya_hal     foreign_markets
   (IBB)     (Open-Meteo) (Frankfurter) (Playwright)  (FAO/USDA/EU)
        │          │           │          │              │
        └──────────┴───────────┼──────────┴──────────────┘
                               ▼
                     data/raw/*.csv  (idempotent writers)
                               │
                               ▼
                     src/features/feature_builder.py
                     src/data/policy_events.py
                     src/data/demand_features.py
                               │
                               ▼
                  data/processed/feature_matrix.csv
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         baseline.py     advanced.py     farmer.py
        (Linear/RF/XGB) (tuned + SHAP)  (Antalya target)
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                       models/*.joblib
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          predictions     scenario_alerts  prediction_tracker
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                  data/processed/*.csv + .json + .txt
                               │
                               ▼
                       dashboard.py (Streamlit)
                               │
                               ▼
                       Render free tier (HTTPS)
```

### 2.6 Deployment Options and Cost Implications

| Option | Setup cost | Monthly cost | When it makes sense |
|---|---|---|---|
| **Render free tier (chosen)** | Zero — `render.yaml` only | $0 (sleeps after 15 min idle) | Capstone, demos, low-traffic public dashboards |
| Render starter | Zero | ~$7 | Always-on, faster cold start |
| Railway / Fly.io | Zero–low | $5–10 | Comparable; Fly.io edge regions help LATAM/EU split audiences |
| Streamlit Community Cloud | Zero | $0 | Cleanest path for Streamlit specifically; less control over env |
| Self-hosted on a VPS (Hetzner CX11) | ~1 hour Nginx + systemd | ~$5 | When you need persistent storage > free tier, custom domain, or non-HTTP services |
| AWS App Runner / Cloud Run | Higher (IAM, CI hooks) | ~$10–25 + traffic | Once you outgrow a single dyno or need autoscaling |
| Kubernetes (EKS / GKE) | High | $70+ baseline | Multi-service, multi-team — overkill for this scope |

We deploy to **Render free tier** because the dashboard is read-only, low-traffic, and cold starts are acceptable for a graded artifact. The model artifacts live in the repo (~14 MB joblib total), so cold starts only need to pull the container image, not retrain. If the project graduated to operational use, the recommended next step would be **Render starter ($7/mo)** for always-on, then **Cloud Run** if multi-region demand emerged.

### 2.7 Security Considerations

- `.env`, `models/`, and large `data/raw` artifacts handled via `.gitignore`.
- Dashboard is public read-only; no credentials are stored or required.
- All third-party APIs are public and read-only — no credentials in the repo.
- The Antalya scraper rate-limits itself (1 second between dates) to respect the source.
- No PII processed; only aggregate market data.

### 2.8 Observability

- `data/processed/refresh_log.csv` records every collector run: timestamp, source, rows before/after, status, error message.
- The dashboard renders a **Data Freshness Banner** on every page load, surfacing per-source staleness with traffic-light icons.
- `data/processed/prediction_history.csv` and `accuracy_report.csv` give a continuous read on live forecast quality.
- GitHub Actions emails on failure; the workflow's last-30-line tail is dumped on `::error::` for fast triage.

---

## 3. Software Testing

### 3.1 Testing Strategy

| Level | Where | Tool | What it asserts |
|---|---|---|---|
| **Unit** | `tests/test_*.py` | pytest | Pure-function correctness: breakeven math, season phase mapping, alert thresholds, parsers |
| **Integration / smoke** | `tests/test_pipeline_smoke.py` | pytest + tmp dirs | Pipeline stages chain correctly given a tiny synthetic input |
| **Contract (data shape)** | `tests/test_schemas.py` | pytest | Collectors return DataFrames with the documented columns and dtypes |
| **End-to-end (manual)** | `streamlit run dashboard.py` | manual | Every dashboard page loads, every metric and chart renders without exception |
| **Continuous (live data)** | `.github/workflows/daily-update.yml` | GitHub Actions | The full pipeline runs daily on real new data; failures alert maintainers |

### 3.2 Why pytest

- Standard for Python; available out-of-the-box on every developer machine.
- Plays well with the GitHub Actions matrix.
- Test-discovery convention removes per-test boilerplate.
- Fixtures keep the integration tests' synthetic inputs DRY.

### 3.3 Coverage Goals and Quality Gate

- **Quality gate:** `pytest tests/` must pass and exit zero before any merge to `main`.
- **Coverage target:** 70% on `src/` (configured in `.ctoc/settings.yaml`). The dashboard module is excluded — Streamlit code is best validated by manual smoke testing against the deployed app.

### 3.4 What we deliberately do NOT test

- **Live API responsiveness** of İBB, Open-Meteo, Frankfurter, and Antalya municipality. Those checks belong to monitoring (the daily Action effectively serves as a contract canary), not to the unit-test suite.
- **Numerical equality of model predictions** across runs. Tree models are deterministic given a seed, but tiny floating-point differences across CPUs would create flaky tests. We assert on metric ranges (MAE under threshold) instead.
- **Streamlit-rendered HTML.** Streamlit has its own visual-regression story; we validate by manual inspection during the sprint demo.

### 3.5 CI/CD

The repository uses **three GitHub Actions workflows**:

- `.github/workflows/tests.yml` — runs `pytest tests/` on every push to `main` and on every pull request. Acts as the merge-blocking quality gate.
- `.github/workflows/daily-update.yml` — schedule `0 5 * * *` (05:00 UTC = 08:00 Turkey). Pulls fresh data, retrains, commits artifacts. Acts as a continuous live-data smoke test for the entire pipeline.
- `.github/workflows/weekly-update.yml` — broader weekly enrichment of slower-moving sources (foreign markets, policy events).

Pull requests are blocked from merging if the test workflow fails.

### 3.6 Manual Testing Protocol (per sprint demo)

Before each sprint review the following is exercised live in the deployed dashboard:

1. Open the deployed dashboard URL.
2. Verify the freshness banner shows green or expected-yellow.
3. Visit every page in turn — Farmer Panel, Overview, Price Analysis, Weather & Environment, Market & Policy, Demand & Trends, Model Results, Forecasts & Alerts.
4. On Farmer Panel: change the cold-storage slider and confirm the net-gain metric updates.
5. On Forecasts & Alerts → Forecast Tracking: switch horizons and confirm the Predicted-vs-Actual chart redraws.
6. Confirm at least one prior daily-update GitHub Actions run is green.

### 3.7 Defect Handling

Defects are tracked on the Trello board's **Bugs** column. Each gets:

- A reproducible scenario (URL, page, inputs).
- A severity tag (P0 blocks demo, P1 affects accuracy, P2 cosmetic).
- A linked commit when fixed.

Two notable past defects worth documenting:

- **`avg_price` NaN bug (Apr 3, 2026)** — the IBB API returned blank min/max strings for some 2007 rows. Fix: cast and coerce in `hal_prices.py`, fall back to row drop. Captured in commit `32ff34c`.
- **pandas FutureWarning on `M` resample (Apr 18, 2026)** — `resample("M")` is deprecated; switched to `resample("ME")`. Commit `00f2595`.

---

## 4. Compliance with Capstone Rubric

| Rubric criterion | Where in repo |
|---|---|
| Software repository with developed code, documented | This repo, [README.md](./README.md), [CLAUDE.md](./CLAUDE.md) |
| Link to deployed version | [README.md](./README.md) — Render URL |
| Up-to-date task board | Trello link in [README.md](./README.md) |
| Design and testing document | This file |
| Software / architecture patterns used and reasons | §2.2 of this file |
| Testing methods used and reasons | §3 of this file |
| Software engineering methodology and CI/CD | Sections §2.1 and §3.5; see `.github/workflows/` and [SPRINTS.md](./SPRINTS.md) |
| Initial list of user stories | [USER_STORIES.md](./USER_STORIES.md) |
| ≥3 sprints with planning and demos | [SPRINTS.md](./SPRINTS.md) |
| Recorded demonstration | Submitted via Quantic dashboard separately |
| Repository shared with `quantic-grader` | See setup note in [README.md](./README.md) |
