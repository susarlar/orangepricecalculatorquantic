# Orange Price Predictor (Finike, Turkey)

## Project Overview
Machine learning system that forecasts Finike orange wholesale prices 7–90 days ahead. Built as the MSc Software Engineering Capstone for Quantic School of Business and Technology.

The system fuses wholesale market prices (Hal), weather, satellite NDVI, FX rates, competitor-country supply, and policy events into a feature matrix, trains XGBoost/LightGBM/quantile/ensemble models, and serves predictions through a Streamlit dashboard targeting farmers, traders, exporters, and analysts.

## Tech Stack
- **Language:** Python 3.11
- **ML / Data:** pandas, NumPy, scikit-learn, XGBoost, LightGBM, statsmodels, SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web:** Streamlit dashboard, deployed on Render
- **Data sources:** İBB Hal API (wholesale prices), Open-Meteo (weather), Sentinel-2 (NDVI), TCMB (FX), Google Trends, FAO/USDA-FAS
- **News + LLM:** Google News RSS → DeepSeek `deepseek-chat` (OpenAI-compatible client) for structured event extraction (sentiment, event type, magnitude, confidence)
- **CI/CD:** GitHub Actions (daily data refresh + model retrain, weekly enrichment)
- **Persistence:** joblib model artifacts, CSV feature store

## Project Structure
```
orangepricepredictor/
├── data/                  # Datasets (gitignored where >10 MB)
│   ├── raw/               # API/scrape outputs
│   └── processed/         # Feature matrix, predictions, model results
├── docs/                  # Project documentation
│   └── data_sources.md    # Data sources, features, priorities
├── notebooks/             # Jupyter exploration
│   └── 01_eda.ipynb
├── src/                   # Application source
│   ├── alerts/            # Scenario alert rules (frost, drought, FX shocks)
│   ├── data/              # Collectors per source
│   ├── features/          # Feature engineering
│   ├── models/            # Baseline, advanced, farmer-facing models
│   ├── utils/             # Shared helpers
│   ├── auto_refresh.py    # Idempotent daily refresh entry point
│   ├── pipeline.py        # CLI: collect / features / train / alerts
│   ├── prediction_tracker.py  # Tracks live prediction accuracy
│   └── config.py          # Constants, region, thresholds
├── tests/                 # pytest suite
├── models/                # Saved .joblib model artifacts
├── reports/               # Generated figures and HTML reports
├── plans/                 # CTOC functional/implementation/execution plans
├── dashboard.py           # Streamlit application
├── USER_STORIES.md        # Capstone backlog
├── DESIGN_AND_TESTING.md  # Architecture + testing decisions
├── SPRINTS.md             # Sprint records and retrospectives
├── render.yaml            # Render deployment config
└── requirements.txt
```

## Capstone Methodology (Iron Loop)
1. **Plan** → user story refinement and sprint planning
2. **Code** → implementation with quality gates
3. **Test** → unit + integration + smoke tests
4. **Review** → architecture, security, code review
5. **Ship** → push to Render, update Trello task board

## Quality Gates
- Public functions have docstrings
- `pytest tests/` must pass before merge
- Data pipeline steps are idempotent and reproducible
- Model metrics tracked in `data/processed/model_results.csv`
- All UI strings, comments, docs in English

## Commands
```bash
# Install
pip install -r requirements.txt

# Tests
pytest tests/ -v

# Pipeline
python -m src.pipeline --collect
python -m src.pipeline --features
python -m src.pipeline --train
python -m src.pipeline --alerts

# Dashboard (local)
streamlit run dashboard.py

# Daily refresh (used by GitHub Actions)
python -m src.auto_refresh --full
```

## Conventions
- snake_case for Python files and functions
- All user-facing strings, comments, and docstrings in **English**
- Notebooks: restart kernel and run-all before committing
- Never commit raw data files larger than 10 MB
- Model artifacts in `models/`, never in `src/`

## Capstone Submission Notes
- Repository must be shared with the GitHub user `quantic-grader`
- Trello task board link is in `README.md`
- `DESIGN_AND_TESTING.md` covers architecture decisions and testing approach
- `USER_STORIES.md` is the prioritized product backlog
- `SPRINTS.md` documents the three+ delivered sprints
