---
type: vision
status: decomposed
decomposed_at: "2026-05-19T09:44:04.538Z"
---

# Vision: Wire DeepSeek News into Market & Policy Tab

## Status
- Created: 2026-05-19T00:00:00Z
- Last Updated: 2026-05-19T00:00:00Z
- Progress: 5/5 phases complete
- Status: ready

## Phase 1: Problem Discovery
### Problem Statement
✓ The Market & Policy tab in `dashboard.py` shows only a hand-curated 36-row event list from `policy_events.csv` whose last entry is dated 2025-01-01, so it looks stale to any grader or domain user who opens the dashboard in 2026. Meanwhile the daily refresh pipeline (`src/auto_refresh.py:251-269` calling `src/data/news.py:refresh_news()`) is already wired to pull Google News, classify each article with DeepSeek `deepseek-chat`, and write per-article rows to `data/raw/news_events.csv` plus daily aggregates to `data/raw/news_features.csv`. Those CSVs are completely invisible to the UI — `load_data()` in `dashboard.py:34-118` never references them. The result is a live, classified, English-language news feed that the dashboard does not surface.

### Target User
✓ Two audiences: (1) the American Quantic capstone graders and the `quantic-grader` GitHub user, who need to see fresh, English news evidence inside the dashboard within two minutes of opening it; (2) Su Sarlar herself plus domain users (farmers, traders, exporters, analysts in the Finike corridor) who want recent market-moving news alongside the curated long-horizon event list.

### Problem Severity
✓ Medium-high for capstone evaluation. The freshness gap is the single most visible "this project is stale" signal on the Market & Policy page — graders will read 2025-01-01 as a red flag. Severity is bounded though: prices, weather, FX and forecasts are all up to date, so this is a UX/credibility fix, not a system failure.

## Phase 2: Value Proposition
### Success Criteria
✓ A grader opening the deployed Streamlit app, navigating to "Market & Policy", and scrolling the tab can verify all of the following in under two minutes without touching code:
1. A new "Recent News (DeepSeek-classified)" section is visible alongside the existing "Policy and Event Timeline" and "Event List".
2. At least one row in that section is dated within the last 30 days and renders in English (the `llm_summary` field already produces a one-sentence English summary up to 140 chars).
3. Each row shows date, English summary, event type (human-readable label), sentiment, magnitude (1-3), and confidence.
4. If `news_events.csv` is missing or empty (e.g., `DEEPSEEK_API_KEY` not configured in that environment), the section degrades gracefully with an informative message instead of crashing the page.

### Impact Scale
✓ Single dashboard section, single CSV load, ~2 helper functions. Affects one tab on one page. No effect on training, feature matrix, models, alerts, or the daily refresh job. Estimated effort: well under one engineering day.

## Phase 3: Scope Definition
### Minimum Viable Scope
✓ In scope:
- Add `news_events_path = RAW_DIR / "news_events.csv"` (and optionally `news_features.csv`) loading to `load_data()` in `dashboard.py`, parsing the date column and tolerating missing files.
- Add a new subsection on the Market & Policy tab (tab2) titled "Recent News (DeepSeek-classified)" rendered below the existing Event List.
- Sort news by date descending, show the most recent N rows (default 25, configurable via a small `st.slider` or hardcoded constant), and display columns: Date, Summary (English `llm_summary`), Type, Sentiment, Magnitude, Confidence.
- Use the existing `event_type_to_human` / `magnitude_to_human` helpers where the vocab overlaps; fall back to title-casing the raw string otherwise.
- Graceful fallback `st.info("No DeepSeek-classified news yet. Set DEEPSEEK_API_KEY and run the daily refresh.")` when the CSV is absent or empty.
- One-line README or `docs/data_sources.md` note pointing graders at the new section.

### Explicit Exclusions
✓ Out of scope (deferred / not this vision):
- No changes to `src/data/news.py`, no prompt rewrites, no schema migrations to `news_events.csv`.
- No new DeepSeek API calls from the dashboard — the dashboard only reads the CSVs the daily job already produces.
- No retraining, no addition of news features to the feature matrix or models (those already exist via `news_features.csv` and are out of scope for this UI change).
- No translation layer (the `llm_summary` field is already English by construction).
- No merging of `news_events.csv` into the existing `policy_events.csv` timeline scatter plot; the curated long-horizon events and the rolling LLM news feed stay visually separate.
- No new tab, no top-level navigation change, no auth, no write paths.
- Setting `DEEPSEEK_API_KEY` in the deployment environment is treated as a one-line configuration prerequisite the user owns, NOT a code deliverable inside this vision.

### Dependencies
✓ Hard dependencies (already satisfied or owned outside this work):
- `src/auto_refresh.py` and `src/data/news.py` already produce `data/raw/news_events.csv` and `data/raw/news_features.csv` on every daily run when `DEEPSEEK_API_KEY` is set.
- `DEEPSEEK_API_KEY` must be present in the environment running the daily job (locally and on Render) for fresh rows to appear. This is a configuration step, not a coding task.
- Existing dashboard helpers `event_type_to_human`, `direction_to_human`, `magnitude_to_human` are reusable for label rendering.

## Phase 4: Risk Assessment
### Failure Modes
✓ 1. `news_events.csv` exists locally for Su but not in the Render deployment if the secret isn't set there → grader sees the empty-state message instead of fresh rows. Mitigation: explicit fallback copy that names the missing key. 2. Schema drift: `news.py` evolves and renames a column → dashboard crashes. Mitigation: defensive column access (`.get` / `if col in df.columns`). 3. Date parsing inconsistency between `news_events.csv` and the existing event timeline. Mitigation: pass `parse_dates=["date"]` and coerce to UTC-naive like other loaders. 4. Performance: if `news_events.csv` grows to tens of thousands of rows over months, naive rendering may slow the tab. Mitigation: head-N slice before display.

### Unknowns
✓ Whether `DEEPSEEK_API_KEY` is configured on Render today (likely not — local refresh log shows it skipped). The Render env var is a deploy-side action outside the code change. Also unknown: exact column names finalized in the latest `news_events.csv` schema — should be verified against `src/data/news.py` before writing the loader code, not before approving the vision.

### Assumptions
✓ 1. The `llm_summary` field is reliably English (the prompt enforces "English, max 140 chars"). 2. Graders will judge freshness primarily by the most-recent row's date, not by row count. 3. Showing the curated 36-row event list AND the LLM news feed side-by-side is more credible than replacing one with the other — the curated list provides multi-year historical context, the news feed provides rolling freshness. 4. The user does not want auto-merging into the existing scatter plot in this iteration (cleaner separation reads better).

## Phase 5: Summary
## Vision: Wire DeepSeek News into Market & Policy Tab

**In one sentence:** American capstone graders and Finike domain users can see fresh, English, DeepSeek-classified market news on the dashboard by loading the already-produced `data/raw/news_events.csv` into a new section on the Market & Policy tab.

**The problem:** The Market & Policy tab shows only a hand-curated 36-row event list whose last entry is 2025-01-01, which reads as stale in 2026, even though the daily pipeline already produces classified English-language news in `news_events.csv` that the dashboard never loads.

**For whom:** (1) Quantic American graders and the `quantic-grader` GitHub user who need an at-a-glance freshness signal; (2) Su Sarlar plus farmers, traders, exporters, and analysts in the Finike orange corridor who want recent market-moving news next to the long-horizon event list.

**Success looks like:** A grader opens the deployed Streamlit dashboard, clicks "Market & Policy", and within two minutes confirms a "Recent News (DeepSeek-classified)" section showing at least one English-language news row dated within the last 30 days, with type / sentiment / magnitude / confidence columns rendered as human-readable labels, and a graceful fallback message when the CSV is missing.

**What we are building:** A new subsection in `dashboard.py` (Market & Policy, tab2) that loads `news_events.csv` (and optionally `news_features.csv`) via `load_data()`, sorts by date descending, slices the top ~25 rows, and renders a labeled table using existing humanization helpers. Roughly two new helpers and one extended loader path.

**What we are NOT building:**
- No changes to `src/data/news.py`, no prompt edits, no schema migration.
- No new DeepSeek API calls from the dashboard.
- No retraining, no new model features, no changes to the feature matrix.
- No merging of LLM news into the existing event scatter plot.
- No translation layer (summary is already English).
- Configuring `DEEPSEEK_API_KEY` on Render is a one-line ops prerequisite owned by Su, not a code deliverable here.

**Key risk:** `DEEPSEEK_API_KEY` is not set in the Render environment, so the deployed dashboard shows the empty-state fallback instead of fresh rows on submission day. Specific mitigation: write the fallback copy to name the missing key explicitly, and add a one-line note to README / `docs/data_sources.md` so the grader and Su both know what to set.

**RICE Score:**
- Reach: M — every user who opens the Market & Policy tab, with graders being the load-bearing audience.
- Impact: H — directly removes the most visible "stale dashboard" signal during capstone evaluation.
- Confidence: H — all required artifacts (news CSV producer, helpers, tab structure) already exist.
- Effort: L — single tab section, one CSV load, ~2 helpers, well under one engineering day.

**Assumptions to test:**
- `llm_summary` is reliably in English across the latest article batch.
- The current `news_events.csv` schema in `src/data/news.py` matches what we read.
- Render deployment env can have `DEEPSEEK_API_KEY` set before grader review.

## Discussion History
### 2026-05-19
Q: Initial seed parsing
A: Seed had two clauses. Clause 1 ("automate news retrieval") is already shipped — `auto_refresh.py:251-269` calls `refresh_news()` daily; the only gap is the unset `DEEPSEEK_API_KEY` env var, which is a configuration step, not a code change. Clause 2 ("incorporate them into the dashboard with DeepSeek") is the actual remaining work: wire `news_events.csv` into `dashboard.py`'s Market & Policy tab. Scoped accordingly with grader-verifiable success criteria and explicit exclusions to prevent scope creep into prompts, schemas, or retraining.
