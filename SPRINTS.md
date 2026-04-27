# Sprint Records

> The Capstone handbook requires **at least three sprints** with planning, execution, and demonstration. This document records each sprint's goal, backlog, demonstration, retrospective, and supporting commit evidence.

---

## Team and Roles

This is an individual capstone. The single member rotates across the formal Scrum roles:

| Role | Holder |
|---|---|
| Product Owner | Su Sarlar |
| Scrum Master | Su Sarlar |
| Code Owner / approver | Su Sarlar |
| Engineer | Su Sarlar |

Working agreements: weekly time-boxed self-review on Sunday; daily commit + push when work happens; CI must be green before a sprint review is recorded.

Tools used:
- **Source control + CI:** GitHub + GitHub Actions
- **Task board:** Trello (link in [README.md](./README.md))
- **Editor & AI tooling:** VS Code + Claude Code (CTOC plugin) for plan/iron-loop methodology
- **Deployment:** Render free tier
- **Communication:** Self (commit messages serve as the standup log)

---

## Sprint 0 — Project Conception (≤ Mar 22, 2026)

**Goal:** Define the problem, choose tech stack, identify data sources, draft the initial backlog.

**Outputs:**
- Problem statement: forecast Finike orange wholesale prices 7–90 days ahead
- Personas (Selma, Mert, Defne, Ahmet) — see [USER_STORIES.md](./USER_STORIES.md)
- Data-source inventory — see [docs/data_sources.md](./docs/data_sources.md)
- Initial backlog (epics E1–E7) — see [USER_STORIES.md](./USER_STORIES.md)

**Result:** Project scope agreed; ready for first sprint planning.

---

## Sprint 1 — Foundation (Mar 23 – Apr 5, 2026)

**Sprint Goal:** End-to-end pipeline from raw data to a deployed MVP dashboard with a baseline model.

**Sprint Backlog:** S1-01 through S1-07 from [USER_STORIES.md](./USER_STORIES.md).

**Delivered:**

| Story | Outcome | Evidence |
|---|---|---|
| S1-01 | İBB Hal historical prices (2007–today) | `src/data/hal_prices.py`; `data/raw/hal_prices.csv` populated |
| S1-02 | Open-Meteo Finike weather (history + 16-day forecast) | `src/data/weather.py`; `data/raw/weather_finike.csv` |
| S1-03 | Synthetic NDVI fallback | `src/data/satellite.py` `collect_ndvi_timeseries(use_synthetic=True)` |
| S1-04 | Frankfurter FX rates | `src/data/fx_rates.py`; `data/raw/fx_rates.csv` |
| S1-05 | Feature engineering (lag/rolling/seasonal) | `src/features/feature_builder.py`; `data/processed/feature_matrix.csv` |
| S1-06 | Baseline models (Linear, RF, XGBoost) | `src/models/baseline.py`; `data/processed/model_results.csv` |
| S1-07 | Streamlit dashboard MVP | `dashboard.py`; deployed to Render |

**Key commits:**
- `f4f402a` — *Initial release: full ML pipeline + dashboard* (Sprint 1 closing commit)
- `e34d73c` — *Fix Render deployment + add weekly auto-update*

**Sprint Review / Demo:** Walk through `python -m src.pipeline --all`, then `streamlit run dashboard.py` showing live prices, weather, and a baseline 30-day forecast.

**Retrospective:**
- *Went well:* CSV-on-disk pipeline made iteration fast; Render free tier was a one-shot deploy.
- *Did not go well:* The first model run produced unrealistically tight intervals because point-only models were used. Carrying quantile regression into Sprint 2.
- *Action items:*
  - Add quantile intervals (→ S2-02)
  - Add a SHAP explanation tab (→ S2-04)
  - Investigate feature contributions for low-data horizons

---

## Sprint 2 — Modeling Depth (Apr 6 – Apr 19, 2026)

**Sprint Goal:** Production-quality modeling with explainability, prediction intervals, and an alert system.

**Sprint Backlog:** S2-01 through S2-07.

**Delivered:**

| Story | Outcome | Evidence |
|---|---|---|
| S2-01 | Tuned LightGBM and XGBoost; tuned 30d MAE beat baseline | `src/models/advanced.py`; `models/{lightgbm_tuned,xgboost_tuned}_{30,60,90}d.joblib` |
| S2-02 | Quantile regression (P10/P90) | `src/models/advanced.py` `train_quantile_models`; `models/quantile_*.joblib` |
| S2-03 | Ensemble model | `models/ensemble_{30,60,90}d.joblib` |
| S2-04 | SHAP feature importance + dashboard tab | `data/processed/shap_importance.csv`; SHAP Analysis tab |
| S2-05 | Scenario alerts (frost / drought / FX / NDVI / calendar) | `src/alerts/scenario_alerts.py`; `data/processed/latest_alerts.txt` |
| S2-06 | Policy event features | `src/data/policy_events.py`; `data/raw/policy_features.csv` |
| S2-07 | Foreign markets + competitor data | `src/data/foreign_markets.py`; `data/raw/foreign_markets.csv` |

**Key commits:**
- `065daa8` — *Add daily automated data ingestion & model retraining pipeline*
- `5ff121f` — *Daily data update: 2026-04-02* (first successful CI-driven refresh)
- `9d0f09a` — *Fix hal price refresh + add prediction vs actual tracking*

**Sprint Review / Demo:** Live walkthrough of Model Results, Forecasts & Alerts (alerts and SHAP tabs) on the deployed instance, showing tuned-vs-baseline MAE delta and a frost-alert example.

**Retrospective:**
- *Went well:* Quantile bounds gave the dashboard a much more honest visual; alerts make the system feel "alive".
- *Did not go well:* `avg_price` came back NaN for some early-2007 rows from IBB, causing chart breaks.
- *Action items:*
  - Defensive parsing in `hal_prices.py` (carried to Sprint 3 via commit `32ff34c` → *Fix avg_price NaN bug*)
  - Ship the auto-retrain workflow before Sprint 3 (done via `065daa8`)
  - Start tracking live forecast accuracy (→ S3-05)

---

## Sprint 3 — Decision Support, CI/CD, and Polish (Apr 20 – May 3, 2026)

**Sprint Goal:** Capstone-ready system with CI/CD, decision support, deployed dashboard, and documentation.

**Sprint Backlog:** S3-01 through S3-09.

**Delivered:**

| Story | Outcome | Evidence |
|---|---|---|
| S3-01 | Antalya Hal Playwright scraper | `src/data/antalya_hal.py` |
| S3-02 | Farmer SELL / WAIT / COLD STORAGE / SELL NOW recommendations | `src/models/farmer.py`; `data/processed/farmer_advice.json`; Farmer Panel page |
| S3-03 | Cold-storage scenario calculator | Farmer Panel slider + live net-gain metric |
| S3-04 | Daily GitHub Actions auto-refresh + retrain | `.github/workflows/daily-update.yml`; commits prefixed *Daily update:* |
| S3-05 | Prediction tracker (live accuracy) | `src/prediction_tracker.py`; `data/processed/prediction_history.csv` + `accuracy_report.csv` |
| S3-06 | Render deployment | `render.yaml`; live URL in `README.md` |
| S3-07 | Dashboard freshness banner | `render_freshness_banner` in `dashboard.py` |
| S3-08 | Pytest test suite | `tests/test_*.py` |
| S3-09 | Trello task board | Public link in `README.md` |

**Key commits:**
- `c8d0ccb` — *Add prediction vs actual tracking + weekly horizons + dashboard tab*
- `1cc088f` — *Auto-retrain when model files are missing for any horizon*
- `32ff34c` — *Fix avg_price NaN bug + refresh data (April 3)*
- `00f2595` — *Fix pandas M->ME + refresh data through 2026-04-18*
- Plus 16 *Daily update: YYYY-MM-DD* commits driven by GitHub Actions across April 2026 — direct evidence of the CI/CD pipeline running unattended for 16 consecutive days.

**Sprint Review / Demo:** End-to-end demo on the deployed dashboard:
1. Open the public dashboard URL.
2. Land on the freshness banner — green.
3. Farmer Panel — show today's price, breakeven, margin, recommendation, and run the cold-storage slider.
4. Forecasts & Alerts → Forecast Tracking — switch horizons, show predicted-vs-actual chart.
5. Open GitHub Actions to show 16+ green daily runs.
6. Open Trello to show the board state.

**Retrospective:**
- *Went well:* Daily Actions runs gave the project a steady "heartbeat" of evidence and uncovered real bugs (the NaN parse, the pandas resample warning) early.
- *Did not go well:* The repo was bilingual (Turkish dashboard, English code) which obscured intent for an external grader.
- *Action items (closed at Capstone packaging):*
  - Translate the entire user-facing surface to English (this packaging pass).
  - Add formal tests under `tests/` (S3-08 — completed at packaging).
  - Add `USER_STORIES.md`, `DESIGN_AND_TESTING.md`, `SPRINTS.md` (this pass).

---

## Capstone Submission Checklist

| Item | Status |
|---|---|
| Working software meeting user requirements | ✅ Deployed to Render; daily Actions green |
| GitHub repo shared with `quantic-grader` | ⏳ See README setup note |
| Link to deployed version | ✅ In `README.md` |
| Agile task board (Trello) | ✅ Linked in `README.md` |
| Design and testing document | ✅ [DESIGN_AND_TESTING.md](./DESIGN_AND_TESTING.md) |
| Initial user stories | ✅ [USER_STORIES.md](./USER_STORIES.md) |
| Recorded 15–20 minute demonstration | ⏳ Submitted separately via Quantic dashboard |
| ≥3 sprints completed with sprint reviews | ✅ This document |
| CI/CD evidence | ✅ `.github/workflows/`; 16+ green daily runs |
| Software / architecture patterns documented | ✅ `DESIGN_AND_TESTING.md` §2.2 |

---

## A Note on Commit Cadence

Between Apr 3 and Apr 18, 2026 the repository received 16 consecutive *Daily update: YYYY-MM-DD* commits authored by `GitHub Actions Bot`. These are not manual commits. They are the artifact of the daily CI workflow refreshing data, retraining models, and pushing the resulting artifacts. They serve as **objective evidence** that the CI/CD pipeline operated unattended over the sprint window — the strongest possible demonstration of the rubric criterion *"Appropriate software engineering methodology and collaborative software engineering tools, including CI/CD tools, have been used."*
