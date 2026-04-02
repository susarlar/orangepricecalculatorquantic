# Plan: Daily Automated Data Ingestion & Model Training

## Goal
Upgrade the pipeline from weekly to **daily** automated runs — fresh data every day, models retrained, predictions updated, alerts generated.

---

## Current State
- `auto_refresh.py` — refreshes hal prices, weather, FX; generates predictions (runs locally with `--schedule`)
- `pipeline.py` — full pipeline: collect → features → train → alerts (manual)
- `.github/workflows/weekly-update.yml` — GitHub Actions, runs **every Monday** only
- Missing: daily schedule, model retraining in daily flow, error notifications, staleness checks

---

## What Changes

### 1. Daily GitHub Actions Workflow
**File:** `.github/workflows/daily-update.yml`

- **Schedule:** Every day at 05:00 UTC (08:00 Turkey time)
- **Steps:**
  1. Fetch latest hal prices (current + previous month)
  2. Fetch weather (last 90 days + 7-day forecast)
  3. Fetch FX rates (last 90 days)
  4. Update demand & policy features
  5. Rebuild feature matrix
  6. Retrain all models (baseline + farmer)
  7. Generate predictions & alerts
  8. Commit updated `data/`, `models/`, predictions to repo
  9. On failure → notify (GitHub Actions notification or issue creation)

- **Safeguards:**
  - Skip model retrain if no new price data was added (no point retraining on same data)
  - Timeout: 30 minutes max
  - Continue on individual data source failure (don't block everything if weather API is down)
  - Log which sources succeeded/failed in commit message

### 2. Smarter `auto_refresh.py` Updates
- Add `--retrain` flag to trigger feature rebuild + model retrain after data refresh
- Add `--full` flag = `--predict --retrain --alerts` (daily production mode)
- Add data staleness check: warn if any source hasn't updated in >3 days
- Return exit codes: 0 = success, 1 = partial failure, 2 = critical failure

### 3. Retire Weekly Workflow
- Remove or disable `weekly-update.yml` (replaced by daily)
- Or keep weekly as a "full rebuild from scratch" fallback

### 4. Data Freshness Tracking
- Add `data/processed/refresh_log.csv` with columns:
  `[timestamp, source, records_before, records_after, status, error_msg]`
- Dashboard can show "last updated" per source

---

## Out of Scope
- Real-time streaming (overkill for daily commodity prices)
- External notification services (Slack, email) — can add later
- Model drift detection (separate plan)

---

## Success Criteria
- [ ] Pipeline runs automatically every day at 08:00 TR time
- [ ] Fresh predictions available in `data/processed/latest_predictions.csv` daily
- [ ] Models retrained only when new data is available
- [ ] Failed runs don't break the repo (graceful error handling)
- [ ] Commit messages show what was updated and what failed

---

## Implementation Order
1. Update `auto_refresh.py` with `--retrain` and `--full` flags
2. Add refresh logging (`refresh_log.csv`)
3. Create `daily-update.yml` workflow
4. Test with `workflow_dispatch` (manual trigger)
5. Disable or repurpose `weekly-update.yml`
