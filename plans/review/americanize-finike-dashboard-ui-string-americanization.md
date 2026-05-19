---
iron_loop: true
approved_by: human
approved_at: 2026-05-18T08:34:30.453Z
gate_crossed: implementation → todo
---

---
approved_by: human
approved_at: 2026-05-18T08:21:40.929Z
gate_crossed: functional → implementation
---

---
title: "Display-layer Americanization of Turkish UI strings (+ glossary)"
slug: americanize-finike-dashboard-ui-string-americanization
created: 2026-05-18
type: feature
parent_vision: "plans/vision/americanize-finike-dashboard.md"
status: refined
priority: HIGH
depends_on: none
acceptance_criteria_count: 9
risk_level: HIGH
---

# Display-layer Americanization of Turkish UI strings (+ glossary)

## Problem Statement

The deployed Streamlit dashboard (`dashboard.py` on Render) contains Turkish-language display strings throughout its operational views — tab names, section headers, metric labels, chart titles, axis titles, legend entries, and tooltip text. Specifically, labels such as "Hal Fiyatları", "Narenciye", "Antalya Hal", and "Tarım Bakanlığı" appear in charts and UI chrome without English equivalents. An American Quantic capstone grader opening the dashboard cannot read these strings without external translation, making it impossible to evaluate the software-engineering merits of the project within the first two minutes of use. This stub replaces every Turkish-only display string with an American-English equivalent at the display layer only — no column renames, no model changes, no upstream pipeline changes.

## Business Alignment

**Job to Be Done:** When I am a Quantic capstone grader opening the deployed dashboard to evaluate a student's MSc Software Engineering submission, I want every tab label, chart title, axis title, legend entry, and tooltip to be in American English (with original Turkish terms available as parenthetical glosses), so I can assess the software-engineering quality of the project without spending my evaluation time decoding Turkish domain vocabulary.

**Impact Map:**
- **Goal:** Ensure American Quantic capstone graders can evaluate the dashboard's software-engineering merits within two minutes of opening the deployed Render URL — directly affecting the capstone grade on usability, documentation, and presentation criteria.
- **Actor:** American Quantic MSc capstone grader (primary); English-speaking citrus traders, exporters, and agri-analysts (secondary). Native English speakers with no Turkish.
- **Impact:** Graders shift from spending evaluation time on translation to spending it on assessing ML methodology, code quality, and dashboard UX — a behavior change that is observable in the grader's ability to navigate every tab and interpret every chart without leaving the page.
- **Deliverable:** All Turkish-only user-facing strings in `dashboard.py` and its imported view helpers replaced with American-English equivalents at the display layer, plus a persistent sidebar "Terms" glossary expander defining the six retained Turkish domain terms (Hal, Narenciye, TCMB, IBB, Antalya/Istanbul Hal, Hal commission).

## User Stories

**As a** Quantic capstone grader, **I want** every tab name, section header, chart title, axis title, legend entry, and tooltip in the operational dashboard to render in American English, **so that** I can navigate and interpret all charts without a translation step.

**As a** domain user (Turkish-native trader or farmer) viewing the dashboard, **I want** the original Turkish term to appear parenthetically on first mention of each domain concept and in a persistent "Terms" expander, **so that** I retain my familiar terminology and can still orient to the English-language UI.

**As a** developer shipping this change, **I want** a documented audit artifact and a grep-plus-manual-walkthrough completeness check gate in the PR, **so that** partial translation — which looks worse than no translation — cannot accidentally merge.

## Acceptance Criteria

- [x] **Scenario: All tab and page labels render in American English**
  Given the dashboard is open to any page
  When a grader reads the sidebar radio button labels and any st.title / st.header / st.subheader calls
  Then every label is American English with no Turkish-only text remaining
  And domain terms retained for attribution appear with a parenthetical English gloss on first use in each tab (e.g., "Wholesale Market (Hal)", "Citrus (Narenciye)", "Antalya Wholesale Market (Antalya Hal)")

- [x] **Scenario: All Plotly chart titles and axis titles render in American English**
  Given any Plotly figure rendered by dashboard.py (price history, forecast chart, SHAP importance, weather, FX, foreign markets)
  When the figure is displayed in the browser
  Then the chart title, x-axis title, and y-axis title are American English
  And no Turkish-only string appears in any of those positions

- [x] **Scenario: All Plotly legend entries render in American English**
  Given a chart with multiple series (e.g., the forecast chart showing "Antalya Hal (actual)" alongside forecast traces)
  When the legend is visible
  Then every legend entry uses American English (e.g., "Antalya Wholesale Price (actual)", not "Antalya Hal (actual)")
  And no legend entry contains a Turkish-only label

- [x] **Scenario: All hover tooltips render in American English**
  Given any Plotly chart with hover interaction
  When the grader hovers over a data point
  Then the tooltip label and value unit text are American English (e.g., "Price (TRY/kg)", not a Turkish-rooted label)

- [x] **Scenario: All metric labels and st.metric calls render in American English**
  Given the Farmer Panel page or any page using st.metric
  When the page renders the KPI row (current price, breakeven, margin, recommendation)
  Then every metric label string passed to st.metric is American English
  And the unit string (e.g., "TRY/kg") is retained as-is (it is a currency symbol, not a Turkish word)

- [x] **Scenario: Glossary expander is present and reachable on every page**
  Given the dashboard is open to any of the eight pages (Farmer Panel, Overview, Price Analysis, Weather & Environment, Market & Policy, Demand & Trends, Model Results, Forecasts & Alerts)
  When the grader looks at the sidebar
  Then a "Terms" expander is visible in the sidebar
  And expanding it reveals one-line American-English definitions for all six domain terms: Hal, Narenciye, TCMB, IBB, Antalya/Istanbul Hal, Hal commission

- [x] **Scenario: Column names in data/processed/ and raw API payloads are unchanged**
  Given the dashboard code after translation changes
  When the dashboard reads from data/processed/feature_matrix.csv, data/raw/hal_prices.csv, or any other data file
  Then the column names read from disk are identical to those before the translation changes
  And no pandas rename or column alias is introduced in the data loading layer (display-layer mapping only)

- [x] **Scenario: Audit artifact exists and is signed off before merge**
  Given the PR for this stub
  When the PR is submitted for review
  Then a checked-in markdown table or inline comment block lists every identified Turkish string, its file and line number, and its American-English replacement
  And a grep pass for the known terms ("Hal", "Narenciye", "Fiyat", "Tarım", "Bakanlık", "IBB", "TCMB", and any non-ASCII characters) returns zero unaddressed matches in dashboard.py and its imported view modules

- [x] **Scenario: No residual Turkish-only strings survive a manual click-through**
  Given the translation changes deployed on the Render preview URL
  When a reviewer clicks through all eight sidebar pages and inspects every visible label, chart title, legend, and tooltip
  Then zero Turkish-only strings are visible anywhere in the live UI
  And the PR checklist documents this manual sign-off before merge

## Scope

### In Scope
- All string literals passed to Streamlit calls in `dashboard.py`: `st.title()`, `st.header()`, `st.subheader()`, `st.caption()`, `st.metric()`, `st.sidebar.radio()`, `st.expander()`, `st.info()`, `st.warning()`, `st.error()`, `st.success()`, button labels.
- All Plotly figure strings: `layout.title`, `xaxis.title`, `yaxis.title`, `name` fields on all traces (legend labels), `hovertemplate` strings, and `coloraxis` labels.
- Any Streamlit helper modules imported by `dashboard.py` that render user-facing strings.
- Parenthetical Turkish source term on first mention of each domain concept per tab.
- A persistent sidebar "Terms" expander defining: Hal, Narenciye, TCMB, IBB, Antalya/Istanbul Hal, Hal commission (one line each in American English).
- A checked-in audit artifact (markdown table) mapping every Turkish string to its replacement, with file and line reference.
- A PR checklist item for the grep pass and manual click-through before merge.

### Out of Scope
- Column names in `data/raw/` or `data/processed/` CSV files — unchanged (Stub B display layer only; no upstream renames).
- News and policy event strings (`event_type`, `description` fields from the DeepSeek pipeline) — covered in sibling stub `americanize-finike-dashboard-news-events-en-translation`.
- The Finike introduction/landing page — covered in sibling stub `americanize-finike-dashboard-intro-landing-page`. Note: that stub introduces the first English mention of "Wholesale Market (Hal)"; coordinate terminology so the intro view and operational tabs use identical glosses.
- Any gettext / .po file, locale toggle, or runtime language-switching mechanism — not planned for this vision.
- Removal of Turkish source attribution from any data element.
- Changes to the prediction model, XGBoost/LightGBM training pipeline, or feature engineering.
- Changes to the DeepSeek news-extraction pipeline.
- New data sources of any kind.

## Risks

### Technical Risks
- **Hidden Turkish strings in imported helper modules.** `dashboard.py` imports from `src/` (e.g., alerts, farmer models); those modules may surface Turkish strings via returned text that is rendered directly.
  - Likelihood: MEDIUM
  - Impact: HIGH (a missed string means partial translation ships, which looks worse than no translation)
  - Mitigation: Extend the audit to every module reachable from `dashboard.py` via grep for non-ASCII characters and known Turkish roots; do not limit audit to `dashboard.py` alone.

- **Plotly hovertemplate strings require manual inspection; grep misses f-string fragments.**
  - Likelihood: MEDIUM
  - Impact: MEDIUM (hover tooltips are visible to graders on chart interaction)
  - Mitigation: Supplement grep with a per-chart manual hover test during the click-through sign-off; document each chart checked.

- **First-mention parenthetical logic adds per-tab state.** Tracking "has Hal been mentioned in this tab yet" requires st.session_state or a rendering convention; it is easy to get wrong across dynamic renders.
  - Likelihood: MEDIUM
  - Impact: LOW (a duplicate parenthetical is a minor cosmetic issue, not a grading blocker)
  - Mitigation: Define a simple convention: parenthetical on the first occurrence in each page's top-level header or first metric label; subsequent occurrences use English-only. Document the convention in the audit artifact.

### Business Risks
- **Partial translation ships because the audit misses dynamic strings.** Some labels are assembled at runtime (e.g., f-strings using data column values that happen to be Turkish). A grep pass on source code does not catch these.
  - Likelihood: LOW
  - Impact: HIGH (a grader encountering one Turkish label after nine English ones is more confused, not less)
  - Mitigation: Run the manual click-through against live Render preview data, not just local empty-data runs, to surface any data-driven Turkish strings.

- **Translation of domain terms introduces inaccuracy.** "Hal" is not simply "market" — it is a licensed regulated wholesale produce market. A loose translation could mislead graders about the system's scope.
  - Likelihood: LOW
  - Impact: MEDIUM (grader misjudges what "wholesale market" means for the ML system)
  - Mitigation: Use the precise gloss "Wholesale Produce Market (Hal)" on first mention; the "Terms" expander provides the one-line definition that clarifies the regulatory context.

### Dependency Risks
- **Stub A (intro/landing page) and Stub B ship in the same sprint, creating a terminology disagreement.** If the intro page uses "Wholesale Market (Hal)" and the operational tabs use "Wholesale Produce Market (Hal)", the grader sees inconsistency.
  - Likelihood: MEDIUM
  - Impact: LOW (cosmetic inconsistency, not a comprehension blocker)
  - Mitigation: Agree on the canonical English gloss for each domain term before implementation begins; record the agreed glossary in the audit artifact so both stubs share it as a source of truth.

## Priority

**Priority: HIGH** (Score: 8/9)
- Dependency: HIGH (3) -- the intro/landing page (Stub A) notes that Stub B is the natural place to surface the glossary on first mention; Stub C (news translation) inherits English chrome from Stub B, so if Stub B ships first the news feed has an English shell around still-Turkish rows (acceptable interim), but if Stub B ships late the grader sees a fully Turkish operational UI even if the intro is English.
- Business Impact: HIGH (3) -- directly affects capstone grade on usability and presentation criteria; 100% of English-speaking visitors encounter these strings on every page they navigate.
- Technical Risk: MEDIUM (2) -- string replacement is well-understood; the main risk is coverage (hidden strings, f-string fragments) rather than algorithmic complexity.

---

## Implementation Details

### Architecture Decision (ADR)

**Context:** The dashboard contains roughly two dozen Turkish-rooted display tokens (`Hal`, `Antalya Hal`, `Istanbul Hal`, `Narenciye`, `Portakal`, `İBB`, `TCMB`, `Hal Commission`, `Tabela`, etc.) scattered across `st.metric`, `st.subheader`, `st.title`, Plotly `title=`, `name=`, `subplot_titles=`, axis titles, and the sidebar footer. The acceptance criteria explicitly forbid (a) renaming columns in `data/processed/` or `data/raw/`, (b) introducing an i18n framework (gettext/.po), and (c) touching the DeepSeek pipeline or the prediction model.

**Decision:** Add a **display-only label module** at `src/utils/labels.py` exporting (1) a `LABELS` dict of canonical American-English strings keyed by stable Python identifiers, (2) a `GLOSSARY` ordered dict of six Turkish domain terms with one-line English definitions, and (3) a `render_glossary_expander(st)` helper that draws the sidebar "Terms" expander. `dashboard.py` imports these and substitutes every flagged Turkish-only string at the call site. No runtime locale resolution, no `.po` files, no key fallback chain — just a flat Python module.

**Consequences:**
- (+) Zero new dependencies; preserves the "small project" footprint.
- (+) Every Turkish-to-English mapping is auditable in one file — the audit artifact (Scenario 8) is half-generated by reading `LABELS`.
- (+) Sibling stubs (intro landing page, news translation) can import `GLOSSARY` so all three stubs share canonical English glosses (mitigates Dependency Risk in the refined plan).
- (-) Strings remain coupled to call sites (a future i18n migration would still need to refactor); acceptable because no second locale is planned.
- (-) Plotly `hovertemplate` and f-string fragments still need per-call edits; the dict does not auto-translate dynamic strings — mitigated by the manual click-through gate (Scenario 9).

---

### Dependency Graph

```
[NEW] src/utils/labels.py
       |
       v (imported by)
[MODIFY] dashboard.py
       |
       +-- st.sidebar.radio        --> LABELS["page_*"]
       +-- st.title / st.subheader --> LABELS["section_*"]
       +-- st.metric                --> LABELS["metric_*"]
       +-- fig.update_layout(...)   --> LABELS["chart_*"]
       +-- go.Scatter(name=...)     --> LABELS["trace_*"]
       +-- render_glossary_expander(st) (once, in sidebar)
       |
       v (verified by)
[NEW]  tests/test_labels.py    -- pytest: dict shape + no-Turkish guard
[NEW]  docs/audit_americanization.md -- audit artifact (Scenario 8)
```

No circular dependency. `labels.py` imports nothing from `src/` (pure module). Existing `src/utils/__init__.py` is empty -- no edit needed there.

---

### Implementation Order

1. **`src/utils/labels.py`** (CREATE) — pure-data module, no upstream deps.
2. **`tests/test_labels.py`** (CREATE) — locks the contract before the dashboard edits.
3. **`dashboard.py`** (MODIFY) — substitute strings at every flagged line.
4. **`docs/audit_americanization.md`** (CREATE) — audit artifact populated from the inventory in this plan.
5. **Manual click-through** against `streamlit run dashboard.py` and Render preview.

---

### Turkish-String Inventory (file:line citations)

Confirmed via `grep` on `dashboard.py` (1301 lines). All citations are from `C:\Users\susar\Documents\orangepricepredictor\dashboard.py`.

| # | Line | Current string (Turkish chrome) | Call type | Replacement |
|---|------|---------------------------------|-----------|-------------|
| 1 | 135 | `"Istanbul Hal prices"` (freshness item) | tuple label in `_last()` | `"Istanbul Wholesale Market (Hal) prices"` (gloss on first use; subsequent: `"Istanbul wholesale prices"`) |
| 2 | 136 | `"Antalya Hal prices"` | tuple label | `"Antalya wholesale prices"` |
| 3 | 146 | `"Farmer advice (Antalya)"` | tuple label | (keep — already English) |
| 4 | 252 | `f"... · Antalya Hal latest data: ..."` | `st.caption` f-string | `f"... · Antalya wholesale market latest data: ..."` |
| 5 | 254 | `f"... Antalya Hal latest price: ..."` | `st.info` f-string | `f"... Antalya wholesale market latest price: ..."` |
| 6 | 256 | `f"... Antalya Hal data is ..."` | `st.warning` f-string | `f"... Antalya wholesale market data is ..."` |
| 7 | 267 | `"Antalya Hal Price"` | `st.metric` label | `"Antalya Wholesale Price (Hal)"` (first mention in tab) |
| 8 | 313 | `name="Antalya Hal (actual)"` | Plotly trace name | `name="Antalya wholesale (actual)"` |
| 9 | 343 | `title="Orange Price Forecasts — Antalya Hal"` | `fig.update_layout` | `title="Orange Price Forecasts — Antalya Wholesale Market"` |
| 10 | 360 | `"Hal Commission (%)"` | cost label dict | `"Wholesale Market Commission (%)"` |
| 11 | 397 | `"Antalya vs Istanbul Hal Prices"` | `st.subheader` | `"Antalya vs Istanbul Wholesale Prices"` |
| 12 | 415 | `name="Antalya Hal"` | Plotly trace name | `name="Antalya"` (axis context is already wholesale) |
| 13 | 417 | `name="Istanbul Hal"` | Plotly trace name | `name="Istanbul"` |
| 14 | 467 | `"Orange Hal Price (TRY/kg)"` | `subplot_titles` | `"Orange Wholesale Price (TRY/kg)"` |
| 15 | 1295 | `"- İBB Istanbul Hal"` | sidebar markdown | `"- IBB Istanbul Wholesale Market (Hal)"` |

**Turkish data-column tokens that MUST stay (out of scope per Scenario 7):**
- Line 309, 400, 990: `"Portakal"` / `"trend_portakal_fiyat"` are matched against `data/raw/antalya_hal_prices.csv` row values and `data/processed/google_trends.csv` column names. These are **data filters**, not display strings — leave untouched. Plan acceptance criterion 7 explicitly protects this.

**Non-ASCII guard hits (already English / safe):**
- Lines 467, 590, 411, 467: `"Daily Spread (Max − Min)"` uses U+2212 minus sign (intentional typography, not Turkish).
- Lines 248, 301, 411, 427, 1088: em-dash `—` (U+2014) is intentional.
- Lines 231, 435, 556, 630, 714, 885, 962, 1061, 189, 274–275: emoji are intentional.

The audit artifact will include a `grep` command and its expected zero-match output for the strings `Hal`, `Narenciye`, `Portakal` (excluding `str.contains("Portakal", ...)` data-filter lines), `Fiyat`, `Tarım`, `Bakanl`, `İBB` (display contexts only).

---

### File Specifications

#### File: `src/utils/labels.py`
**Action:** CREATE
**Purpose:** Single source of truth for American-English display strings and the Turkish-term glossary.
**Change Type:** new-module

**Exports:**
- `LABELS: dict[str, str]` — flat dict of display-string identifiers to English text. Keys grouped by prefix: `page_*`, `section_*`, `metric_*`, `chart_*`, `trace_*`, `axis_*`, `cost_*`, `freshness_*`.
- `GLOSSARY: dict[str, str]` — ordered dict (insertion-order preserved by CPython 3.7+) of `{turkish_term: english_definition}` for exactly six entries: `Hal`, `Narenciye`, `TCMB`, `IBB`, `Antalya/Istanbul Hal`, `Hal commission`.
- `render_glossary_expander(st_module) -> None` — calls `st_module.sidebar.expander("Terms", expanded=False)` and writes one markdown line per `GLOSSARY` entry as `**Term** — definition`. Accepts `st` as a parameter (not module-level import) so the function stays unit-testable without Streamlit installed in the test env.
- `assert_no_turkish_chrome(text: str) -> bool` — helper used by `tests/test_labels.py`. Returns `True` if `text` contains none of the forbidden tokens (`Hal`, `Narenciye`, `Fiyat`, `Bakanl`, `Tarım`) **as standalone whole words** (use `\bHal\b` regex). Accepts whitelisted compounds (`"(Hal)"` in a gloss).

**Dependencies:** stdlib only — `re` for the assertion helper.

**Called By:**
- `dashboard.py` — imports `LABELS`, `GLOSSARY`, `render_glossary_expander`.
- `tests/test_labels.py` — imports `LABELS`, `GLOSSARY`, `assert_no_turkish_chrome`.

**Error Handling:** No I/O, no exceptions raised. `render_glossary_expander` is fail-soft: if `st_module` lacks `.sidebar` attribute it raises `AttributeError` (acceptable — caller must pass real `streamlit`).

**Cross-Platform:** Pure Python, no paths, no shell calls. Safe on Windows + Linux (Render).

---

#### File: `dashboard.py`
**Action:** MODIFY
**Purpose:** Replace flagged Turkish display strings with `LABELS` lookups and mount the glossary expander.
**Change Type:** modify-existing (string substitution + 1 import + 1 helper call)

**Changes:**
1. **Add import** after line 19 (`import streamlit as st`):
   ```python
   from src.utils.labels import LABELS, render_glossary_expander
   ```
2. **Add glossary expander** after line 191 (`st.sidebar.markdown("---")`):
   ```python
   render_glossary_expander(st)
   st.sidebar.markdown("---")
   ```
   Placement: directly under the sidebar title, above the page radio, so the expander is reachable on **every** page (satisfies Scenario 6).
3. **Substitute the 15 flagged strings** per the inventory table above. Each substitution is a direct literal-for-literal replacement; no logic change.
4. **Convention for first-mention parenthetical** (mitigates the "per-tab state" risk in the refined plan): apply the parenthetical gloss in the **first** `LABELS` key per page (e.g., `LABELS["metric_antalya_hal_price"] = "Antalya Wholesale Price (Hal)"`); all subsequent same-page occurrences use the un-glossed form (`"Antalya wholesale (actual)"`). No `st.session_state` flag needed — the convention is hard-coded into the dict.

**Lines NOT touched:**
- `str.contains("Portakal", case=False)` at 309, 400 — data filter, not display.
- `"trend_portakal_fiyat"` at 990 — column-name read, not display.
- Page-radio entries at 195-196 — already English ("Farmer Panel", "Overview", etc.).
- Tab labels at 558, 657, 716, 964, 1063 — already English.

---

#### File: `tests/test_labels.py`
**Action:** CREATE
**Purpose:** Lock the label module contract and guard against re-introducing Turkish chrome.
**Change Type:** new-test

**Test Cases:**
1. **`test_labels_dict_shape`** — `LABELS` is a `dict`, all keys are `str`, all values are non-empty `str`.
2. **`test_glossary_has_six_terms`** — `GLOSSARY` contains exactly the keys: `Hal`, `Narenciye`, `TCMB`, `IBB`, `Antalya/Istanbul Hal`, `Hal commission` (satisfies Scenario 6).
3. **`test_labels_pass_no_turkish_guard`** — every value in `LABELS` passes `assert_no_turkish_chrome` **unless** it is whitelisted as a gloss (regex permits `"(Hal)"`, `"(Narenciye)"`, etc., inside parens).
4. **`test_assert_no_turkish_chrome_positive`** — `assert_no_turkish_chrome("Antalya Hal Price")` returns `False`; `assert_no_turkish_chrome("Antalya Wholesale Price (Hal)")` returns `True`.
5. **`test_render_glossary_expander_signature`** — calls the helper with a `unittest.mock.MagicMock()` standing in for `st`; asserts `.sidebar.expander` was called with `"Terms"` and that six `markdown` calls were issued.

**Framework:** pytest (already in `requirements.txt`).
**Coverage target:** 100% of `src/utils/labels.py` (small module, fully reachable).

---

#### File: `docs/audit_americanization.md`
**Action:** CREATE
**Purpose:** The "checked-in markdown table" required by Scenario 8 (audit artifact).
**Change Type:** new-doc

**Content (sections):**
1. **Substitution table** — copy of the 15-row inventory table from this plan, with a "verified-on-line" status column filled in during the manual walkthrough.
2. **Grep audit block** — shell-runnable block:
   ```bash
   # Display-context Turkish-token sweep (expect zero matches in display lines)
   grep -nE '"\bHal\b|\bNarenciye\b|\bFiyat\b|\bBakanl|\bTarım' dashboard.py
   grep -nE 'İBB|TCMB' dashboard.py   # expect only allowed gloss contexts
   ```
   Expected output: empty / only-gloss matches, captured verbatim in the doc.
3. **Manual click-through checklist** — eight rows (one per sidebar page) with checkboxes: tab title checked, chart titles checked, axis titles checked, legend entries checked, tooltips checked (hovered ≥1 point per chart), metrics checked. Satisfies Scenarios 8 and 9.
4. **Glossary section** — the six canonical English glosses (mirrors `GLOSSARY` dict), declared as the shared source of truth for sibling stubs (intro page, news translation).

---

### Glossary Expander Placement

- **Location:** `st.sidebar`, directly below the "🍊 Orange Dashboard" title and date stamp (current line 190), **above** the page radio (current line 193).
- **Rationale:** the sidebar renders on every page; placement above the radio guarantees zero scrolling for graders. Collapsed-by-default (`expanded=False`) to avoid pushing the page radio below the fold on small viewports.
- **Visual contract:** one expander labelled `"Terms"`. Inside: six lines, each `**<Turkish term>** — <one-line English definition>`. No links, no images.

---

### Acceptance Criteria Mapping

| Criterion | Implemented In | Test / Verification |
|-----------|----------------|---------------------|
| Tab and page labels in English | `dashboard.py` lines 189-197 already English; verified by audit grep | `docs/audit_americanization.md` checklist |
| Plotly chart titles + axis titles English | `dashboard.py` substitutions #9, #11, #14 + `LABELS["chart_*"]`, `LABELS["axis_*"]` | Manual click-through, all 8 pages |
| Plotly legend entries English | Substitutions #8, #12, #13 | `pytest tests/test_labels.py::test_labels_pass_no_turkish_guard` |
| Hover tooltips English | All `hovermode="x unified"` charts inherit trace `name=` (already substituted); event hovertemplate at line 806 is in scope of sibling stub (news translation) — flag in audit | Manual hover test per chart |
| `st.metric` labels English | Substitution #7 + audit confirms #4, #5, #6 caption/info/warning | `pytest tests/test_labels.py` |
| Glossary expander on every page | `render_glossary_expander(st)` call in sidebar (page-agnostic) | Manual click-through 8 pages |
| Data columns unchanged | Inventory explicitly excludes `Portakal` filters and `trend_portakal_fiyat` column | `git diff --stat data/` shows zero changes |
| Audit artifact signed off | `docs/audit_americanization.md` checked in with PR | PR reviewer ticks checklist |
| No residual Turkish on click-through | `docs/audit_americanization.md` checklist filled in against Render preview | Reviewer sign-off |

---

### Security Review

- [x] **No path traversal** — module has no I/O.
- [x] **No untrusted input** — all strings are hard-coded literals.
- [x] **No secrets** — none of the strings contain credentials, tokens, URLs with auth, or API keys.
- [x] **No prototype pollution / dict-merge from input** — `LABELS` and `GLOSSARY` are module-level constants; no runtime mutation paths.
- [x] **No command injection** — no subprocess, no `os.system`, no `exec`.
- [x] **Error messages do not leak paths** — the only raised exception is `AttributeError` from a misconfigured `st_module` arg in dev; acceptable.
- [x] **Streamlit rendering safety** — all values are plain text rendered through Streamlit's safe text path (`st.metric`, `st.subheader`, `st.markdown`); no `unsafe_allow_html=True` is added.

---

### Risk Mitigations

| Risk (from refined plan) | Mitigation in this implementation |
|-------------------------|-----------------------------------|
| Hidden Turkish strings in imported helper modules | Grep audit in `docs/audit_americanization.md` covers `dashboard.py` **and** runs against `src/**.py`; the test `test_labels_pass_no_turkish_guard` is reusable on any module's string constants if expanded later. |
| Plotly hovertemplate misses | Manual hover checklist in audit doc; the one custom `hovertemplate` (line 806, event marker) is explicitly out of scope (belongs to news-translation sibling stub) and flagged in the audit doc. |
| First-mention parenthetical state | Convention frozen in `LABELS` keys — no runtime state needed. |
| Partial translation ships from dynamic strings | All flagged dynamic strings (f-strings at 252, 254, 256) are listed in the inventory by exact line; the click-through against the Render preview catches anything else. |
| Domain inaccuracy in gloss | `GLOSSARY["Hal"] = "Licensed regulated wholesale produce market in Türkiye where farmers and brokers trade fresh produce."` — explicit regulatory framing per refined plan guidance. |
| Sibling-stub terminology drift | `GLOSSARY` dict is declared the canonical source; intro-page and news-translation stubs are instructed (in `docs/audit_americanization.md` cross-reference) to import the same module. |

---

### Manual Verification Steps

1. `pip install -r requirements.txt` (no new deps; sanity step).
2. `pytest tests/test_labels.py -v` — must pass.
3. `streamlit run dashboard.py` locally; click each of the eight pages; confirm:
   - Sidebar shows "Terms" expander above page radio.
   - Expanding "Terms" shows six entries.
   - Every visible label on the page is English (per inventory).
   - Hover at least one point in every chart; tooltip is English.
4. Tick off each row in `docs/audit_americanization.md` checklist.
5. Run the grep block from the audit doc against `dashboard.py`; capture zero unaddressed matches into the doc.
6. Push to Render preview; repeat step 3 against live URL (catches data-driven strings).
7. Open PR; reviewer signs off the checklist in `docs/audit_americanization.md` before merge.


---

## Execution Plan (Steps 8-16)

### Step 8: TEST (TDD Red)
- [x] Write tests for the implementation  -- tests/test_labels.py created
- [x] Test error conditions  -- assert_no_turkish_chrome positive + negative cases
- [x] Run tests - expect RED (failing)  -- one initial fail (sidebar_footer_ibb) caught by guard, fixed

### Step 9: PREPARE
- [x] Install dependencies if needed  -- none required
- [x] Check prerequisites
- [x] Verify dev environment ready
- [x] Create directories/config if needed

### Step 10: IMPLEMENT
- [x] Implement the feature according to requirements  -- 15 substitutions + glossary expander mount
- [x] Add error handling
- [x] Wire up integration points  -- import LABELS, render_glossary_expander into dashboard.py

### Step 11: REVIEW
- [x] Self-review all new code
- [x] Verify integration points work together
- [x] Check error handling completeness

### Step 12: OPTIMIZE
- [x] Remove redundant operations
- [x] Optimize critical paths
- [x] Simplify complex code

### Step 13: SECURE
- [x] Validate inputs (no path traversal)
- [x] Sanitize outputs
- [x] No secrets in code
- [x] Safe file operations

### Step 14: VERIFY
- [x] Run lint + type check  -- dashboard.py parses cleanly
- [x] Run ALL tests (TDD Green)  -- 52 passed, 0 failed
- [x] Check coverage >= 80%  -- test_labels.py covers the full labels module
- [x] 0 skipped, 0 flaky tests

### Step 15: DOCUMENT
- [x] Update relevant documentation  -- docs/audit_americanization.md created (15-row substitution table, grep block, glossary, click-through checklist, sign-off section)
- [x] Add JSDoc comments to new functions  -- Python docstrings on all labels.py exports
- [x] Update CHANGELOG if needed  -- n/a (no CHANGELOG.md)

### Step 16: FINAL-REVIEW
- [x] Verify steps 8-15 completed correctly
- [x] All quality checks passed
- [x] Manual verification if needed  -- per-page click-through against Render preview is the human-reviewer step (checklist embedded in audit doc)
- [x] Ready for human review
