# Capstone Presentation Guide

> **Goal:** a 15–20 minute recorded demonstration that scores **5** on the Quantic Capstone Presentation Rubric.
> **Format:** screen share + webcam (you on camera, deliverable on screen) with voice-over.
> **You'll cover three things:** the deployed website, the architecture, and the GitHub repository.

---

## Mandatory Setup (do this BEFORE you hit record)

The rubric requires several things that are easy to forget under recording stress. Tick them all first.

| ✅ | Item | Why |
|---|------|-----|
| ☐ | **Government-issued ID** ready to show on camera | Required by the handbook — failing this can void the submission |
| ☐ | Webcam on, face visible, well-lit | Rubric: "students are clearly visible" |
| ☐ | Microphone tested, no background noise | Rubric: "audible" |
| ☐ | Browser at the **deployed Render URL** (logged in if needed) | Rubric: "deliverable operates correctly throughout" |
| ☐ | Browser at **GitHub Actions tab** showing green daily runs | Direct CI/CD evidence |
| ☐ | Browser at **Trello board** | Required by handbook |
| ☐ | Browser at **`docs/architecture.html`** open locally (file:///…) | For the architecture walkthrough |
| ☐ | VS Code or GitHub repo open at the `src/` tree | For brief code walk |
| ☐ | Close Slack / Discord / personal email | Avoid notification interruptions |
| ☐ | Set screen resolution to 1080p, font sizes large enough to read | Rubric: "screen share is clear and legible" |
| ☐ | Recording tool: Zoom (records you + screen in one file) | Handbook recommends Zoom |
| ☐ | Confirm the video destination is **stable** (Google Drive / OneDrive / unlisted YouTube). **No WeTransfer.** | Handbook explicit warning |

> **Recording tip:** record one continuous take. Editing recordings looks unprofessional and the rubric penalises it implicitly under "professional." If you fluff a sentence, just pause and continue — Quantic graders are looking at the substance.

---

## The 18-Minute Script (target middle of the 15–20 range)

This timing is calibrated for an **18-minute** recording. If you're naturally faster, expect ~16 min; if slower, ~20 min. Both pass the rubric.

---

### 🎬 0:00 – 1:00 · Intro + identity check (60 s)

> *"Hello, I'm Sercan Susar, and this is my MSc Software Engineering Capstone for Quantic School of Business and Technology — the **Orange Price Predictor**, an AI-augmented decision-support system for Finike orange prices."*

**On camera:** hold up your government-issued ID for ~3 seconds, name and photo clearly visible. Say:

> *"For verification, here is my government-issued ID."*

**Then state the agenda:**

> *"In the next 18 minutes I'll walk through three things: the deployed website with the user stories it implements, the system architecture, and the GitHub repository including CI/CD evidence."*

---

### 🎬 1:00 – 3:00 · Problem + personas (2 min)

**Stay on camera; share a slide or just speak over a still of the dashboard home page.**

> *"Finike, on the southern Turkish coast, is the country's premium orange-growing region. A grower, a wholesale trader, an exporter, and a ministry analyst all need to know **where prices are heading 7 to 90 days out** — but the data they'd need lives in ten different places: the İstanbul wholesale market API, the Antalya municipality's website, weather APIs, satellite imagery, FX rates, and policy events."*

> *"I built a system that fuses all of those into a single feature matrix, trains an ensemble of gradient-boosted models, and serves daily forecasts and decision recommendations through a public web dashboard. Today I'll demonstrate it through four personas from my user-stories backlog: **Selma**, a Finike farmer; **Mert**, a wholesale trader; **Defne**, an exporter; and **Ahmet**, a ministry analyst."*

> *"Let me show you the deployed deliverable."*

---

### 🎬 3:00 – 10:00 · Live dashboard demonstration (7 min)

**This is the heart of the presentation.** The rubric specifically rewards "thorough, clear and concise demonstration of the user stories ... for a range of user inputs."

**Switch to the deployed Render URL.** Walk through the pages **in this order** — each page maps to specific user stories from `USER_STORIES.md`.

#### Page 1 — Farmer Panel (~2 min) → demonstrates **S3-02, S3-03**

> *"This is the Farmer Panel — the home page for Selma. Today's Antalya Hal price is X TRY per kilogram, the breakeven is Y, and the recommendation is **\<read whatever it shows\>**."*

**Click and explain:**
1. Top KPIs: current price, breakeven, margin, recommendation badge.
2. **Demonstrate the cold-storage slider** — drag it from 0 → 30 → 60 → 90 days. Narrate the live recompute:
   > *"As I move from 30 to 60 days, the storage cost goes from X to Y, the expected sale price comes from the 60-day model, and the net gain or loss updates live. This is the COLD STORAGE decision branch from the user story S3-02."*
3. Forecast chart with P10–P90 intervals — call out the dashed lines and the breakeven line.
4. Antalya vs Istanbul spread chart at the bottom.

#### Page 2 — Overview (~45 s) → demonstrates **S1-01, S1-07**

> *"The Overview page gives any user the bird's-eye view: latest price, the historical min-max band, the 30-day moving average, and the daily spread. The yearly averages and monthly seasonality let me see immediately that prices peak in summer when supply is gone."*

#### Page 3 — Price Analysis (~30 s) → demonstrates **S1-05**

> *"Price Analysis has three tabs — trend with momentum, volatility, and year-over-year overlay. Switching to the YoY tab and selecting 2024, 2025, 2026 lets a trader spot whether this year is tracking the previous one."*
- **Click** the YoY tab. **Multi-select** 2024, 2025, 2026.

#### Page 4 — Weather & Environment (~30 s) → demonstrates **S1-02**

> *"This is where the frost story lives. Selma cares more about the temperature line than any model output, because frost is the single biggest price driver. The bottom-tab scatter shows price-vs-temperature for the whole history."*

#### Page 5 — Market & Policy (~45 s) → demonstrates **S2-06, S2-07**

> *"For Defne the exporter, this page is the most relevant. The first tab overlays USD/TRY against orange prices. The second tab plots historical policy events on the price chart — sanctions, frost events, the 2018 currency crisis — each as a triangle marker. The third tab compares Turkey to its global competitors: Egypt, Spain, South Africa."*
- **Click** the Policy Events tab. **Hover** over one of the triangle markers.

#### Page 6 — Demand & Trends (~30 s)

> *"Google Trends search interest, Ramadan periods shaded green, the input cost index, tourism intensity, and the CPI — these are the demand-side signals."*

#### Page 7 — Model Results (~45 s) → demonstrates **S1-06, S2-01**

> *"This page proves the modeling claim. The best model is highlighted at the top — **\<read MAE/MAPE/R²\>**. The two bar charts compare every model across every horizon. At the bottom, the top-20 features ranked by correlation with the 30-day target — frost-related features and lagged prices dominate, exactly as theory predicts."*

#### Page 8 — Forecasts & Alerts (~1 min) → demonstrates **S2-02, S2-04, S2-05, S3-05**

> *"Four tabs. **Price Forecasts** shows the current 7- to 90-day forecast with intervals. **Forecast Tracking** is the live-accuracy log — predicted versus actual for every prediction whose target date has passed; this is how I prove the model isn't just overfit. **Alerts** is the rule engine — frost, drought, FX, NDVI, calendar — sorted by severity. **SHAP Analysis** is the explainability layer for any single forecast."*
- **Click** through each of the four tabs briefly.

#### Quick freshness banner mention

> *"Notice the banner at the top of every page: it shows today's date and the last refresh of every data source. Right now everything is green, which means the daily CI pipeline ran this morning."*

---

### 🎬 10:00 – 13:00 · Architecture walkthrough (3 min)

**Switch to `docs/architecture.html` open locally in the browser.** Scroll smoothly between sections — don't dwell.

> *"The architecture is documented in this self-contained HTML deck. Let me walk through the three diagrams that matter most."*

**Diagram 1 — System Context (~30 s):**
> *"At the top level: five user personas, ten external data sources, two surfaces — the dashboard and the GitHub-Actions-driven pipeline. The system is read-only from both directions: no orders are placed and nothing is written back to source APIs."*

**Diagram 2 — Data Pipeline (~45 s):**
> *"Internally it's a Pipes-and-Filters pipeline. Every stage reads from disk and writes to disk. This is deliberate — it means any single failure can't break the day's run, every artifact is auditable in the GitHub UI as a CSV diff, and I can re-run any single stage in seconds during development."*

**Diagram 3 — Pattern Map (~45 s):**
> *"The codebase uses nine well-known patterns: Repository for each data source, Strategy for the model families, Adapter for the heterogeneous APIs, Composite for the alert engine, plus DTOs, the Streamlit cache decorator, idempotent writers, and a soft circuit-breaker that lets one source fail without taking down the rest. All of this is documented in `DESIGN_AND_TESTING.md` section 2.2."*

**Diagram 4 — CI/CD Flow (~30 s):**
> *"And this is the daily refresh — at 05:00 UTC, GitHub Actions checks out the repo, installs dependencies, runs the full pipeline, commits the new data and retrained models, and pushes. Render auto-deploys on push, so within minutes the dashboard reflects today's data."*

**Diagram 5 — Sprint Timeline (~30 s):**
> *"And here's the sprint Gantt — three sprints over six weeks, with the daily CI workflow running continuously through Sprints 2 and 3, which is the strongest evidence I can offer that the methodology was actually applied."*

---

### 🎬 13:00 – 16:00 · GitHub repository + CI/CD evidence (3 min)

**Switch to GitHub.** This is where you bank the rubric points for *"appropriate software engineering methodology and CI/CD tools."*

#### Repo overview (~30 s)

**Show:** the repository home page with the README rendered.

> *"Here's the public repository. The README is the landing page — Capstone submission links at the top, then the deliverable summary, the quick-start, and the repository layout. Three live badges across the top: the Tests workflow status, the daily-refresh status, and the deployment target."*

#### Three workflows (~45 s)

**Click:** Actions tab → workflows list.

> *"Three GitHub Actions workflows. **Tests** runs `pytest tests/` on every push and every pull request — it's the merge-blocking quality gate. **Daily refresh** runs the full pipeline every morning at 05:00 UTC. **Weekly update** does the slower-moving sources. Let me show you the daily refresh history."*

**Click:** Daily refresh workflow.

> *"You can see 16+ consecutive green runs through April 2026. These are not manual — they are unattended evidence that the CI/CD pipeline operates correctly day after day."*

**Click:** open one of the *Daily update: 2026-04-XX* commits in the commit log, point at the diff:

> *"And each green run produces a real commit: new rows in `hal_prices.csv`, new rows in `weather_finike.csv`, retrained `.joblib` models. This is the rubric criterion 'Appropriate software engineering methodology and collaborative software engineering tools, including CI/CD tools, have been used' — turned into an audit trail."*

#### Tests workflow (~30 s)

**Click:** Tests workflow → latest run.

> *"And here's the most recent test run — 26 unit tests across config integrity, the breakeven math, the season-phase mapping, the SELL/WAIT/COLD STORAGE decision logic, the alert engine, and a guard test that ensures policy descriptions stay in English. All passing in under four seconds."*

#### Repository layout (~45 s)

**Switch to:** the repo root file listing.

> *"The repo is organized into eight top-level docs and six top-level directories. The four documents that matter for grading are: `README.md` for the submission entry point, `USER_STORIES.md` with the INVEST-format backlog and Gherkin acceptance criteria, `DESIGN_AND_TESTING.md` for the architecture and testing approach, and `SPRINTS.md` documenting all three sprints with retrospectives."*

**Click:** `src/` — show the layered structure briefly.

> *"The application code is in `src/`, organized by responsibility: collectors, features, models, alerts, plus the orchestrators. The Streamlit dashboard lives in `dashboard.py` at the root."*

---

### 🎬 16:00 – 17:00 · Trello task board (60 s)

**Switch to the Trello board.**

> *"The agile task board is on Trello. Five columns — Backlog, Sprint Backlog, In Progress, Review, and Done. Every story from `USER_STORIES.md` exists as a card in the appropriate column, with sprint labels and a link back to the merge commit when it shipped. The handbook requires this evidence and here it is."*

- **Scroll** through the Done column.
- **Open one card** briefly to show the linked commit and acceptance criteria.

---

### 🎬 17:00 – 18:00 · Wrap-up (60 s)

**Back to camera.**

> *"To recap: I built a production-quality forecasting system that integrates ten data sources, runs unattended on a daily CI pipeline, and serves four user personas through a public Streamlit dashboard. Three sprints, 26 passing tests, 16 days of green CI runs, and full documentation."*

> *"All deliverables are in the repository at github.com/susarlar/orangepricecalculatorquantic, the live dashboard is at \<your Render URL\>, and the Trello board is at \<your Trello URL\>. Thank you for reviewing my Capstone."*

**End the recording.**

---

## Rubric Self-Check (run through this AFTER recording, BEFORE submitting)

If any row is **No**, re-record before submitting.

### Capstone Presentation Rubric (target: 5)

| Criterion | Yes / No |
|---|---|
| Thorough, clear, concise demo of **user stories** for a **range of user inputs** | ☐ |
| Deliverable operates correctly **throughout** the demo (no errors, no broken tabs) | ☐ |
| Presentation is **professional**; screen share is **clear and legible** | ☐ |
| Student is **clearly visible and audible** throughout | ☐ |
| Government-issued ID shown to camera | ☐ |
| Recording is between **15 and 20 minutes** | ☐ |

### Capstone Project Rubric (target: 5) — these come from the repo, not the video, but check them while you're at it

| Criterion | Yes / No |
|---|---|
| Repository contains all developed code, documented | ☐ |
| Repository linked to deployed version (URL in README) | ☐ |
| Up-to-date Trello board, all stories present | ☐ |
| Detailed design and testing document (`DESIGN_AND_TESTING.md`) | ☐ |
| CI/CD tools demonstrated (`.github/workflows/`) | ☐ |
| Repository shared with `quantic-grader` GitHub user | ☐ |
| README points to all required artifacts | ☐ |

---

## Common Pitfalls (don't lose easy points)

| Pitfall | Fix |
|---|---|
| Forgetting to show the ID | First minute, hold it up for ~3 seconds. Don't skip. |
| Recording is **14:30** or **20:30** | Out of range. Re-record. The rubric is strict on this. |
| Background noise (fan, traffic, kid) | Use a quiet room. Wear headphones with mic. |
| Dashboard error mid-demo (data gap, model load fail) | Run a fresh `python -m src.auto_refresh --full` an hour before recording so everything is loaded. |
| Browser zoom too small / fonts unreadable | Use Ctrl + to zoom each browser tab to 110–125% before recording. |
| Forgetting Trello | It's required by the handbook. Add a 60-second segment as scripted above. |
| Leaving notifications enabled | Slack / email / Teams notifications mid-demo look unprofessional. Quit or set Do Not Disturb. |
| Submitting the recording via WeTransfer | Handbook explicitly warns the link can expire. Use Google Drive / OneDrive / unlisted YouTube. |

---

## Final Pre-Recording Dry Run (≈ 5 min)

Do this once with no recording:

1. Open all four tabs in the right order: Render dashboard → architecture.html → GitHub repo → Trello.
2. Click through the eight dashboard pages once. Confirm none throw an error.
3. Verify the freshness banner is green (run `python -m src.auto_refresh --full` if it isn't).
4. Open the **most recent** green Tests workflow run on GitHub. Bookmark it.
5. Open the **most recent** *Daily update: YYYY-MM-DD* commit. Bookmark it.

Now record, sleep on it overnight, watch it once tomorrow, and submit.

Good luck. 🍊
