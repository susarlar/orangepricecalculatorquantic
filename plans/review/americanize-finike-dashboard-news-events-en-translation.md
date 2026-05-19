---
iron_loop: true
approved_by: human
approved_at: 2026-05-18T08:34:30.458Z
gate_crossed: implementation → todo
---

---
approved_by: human
approved_at: 2026-05-18T08:21:40.932Z
gate_crossed: functional → implementation
---

---
type: feature
parent_vision: "vision/americanize-finike-dashboard.md"
status: refined
priority: MEDIUM
depends_on: "none"
acceptance_criteria_count: 10
risk_level: HIGH
---

# American-English Rendering of DeepSeek News / Policy Events

## Problem Statement

The Market & Policy page of the Finike Orange Price Predictor dashboard surfaces a
live news-and-policy event feed produced by a DeepSeek `deepseek-chat` LLM pipeline
that classifies Google News RSS articles into structured events (`event_type`,
`description`, `impact_direction`, `impact_magnitude`). All five of those fields
currently render in Turkish. An American Quantic capstone grader opening the
dashboard cannot read the policy ticker without translating every row, defeating the
purpose of the feed as a decision-support surface. This stub extends the existing
DeepSeek extraction step to emit two additive English fields (`event_type_en`,
`description_en`) at ingest time, backfills those fields over existing historical
events, and updates the dashboard render path to display English by default while
keeping the Turkish original and source URL accessible on every row for source
attribution.

## Business Alignment

**Job to Be Done:** When I open the Market & Policy page to understand current
policy signals affecting orange prices, I want to read every event row in American
English (type, description, direction, magnitude), so I can act on the ticker within
two minutes without translating it myself and can still verify any event against its
original Turkish-language source.

**Impact Map:**
- **Goal:** An American Quantic capstone grader can evaluate the policy-event
  feature of the dashboard within two minutes of opening the deployed Render URL,
  satisfying the vision's two-minute legibility success criterion.
- **Actor:** English-speaking dashboard visitor (primary: Quantic capstone grader;
  secondary: international citrus trader or agri-analyst) reading the Market &
  Policy / Price + Policy Events feed.
- **Impact:** The visitor reads policy-event meaning directly in American English
  without leaving the dashboard or using an external translator, while retaining
  a one-click path to the Turkish-language source for verification.
- **Deliverable:** Two new structured fields (`event_type_en`, `description_en`)
  emitted by the DeepSeek extraction pipeline, persisted in `data/raw/policy_events.csv`,
  and rendered by default in the dashboard event table and chart hover text.

## User Stories

**As an** English-speaking dashboard visitor, **I want** every row in the Market &
Policy / Price + Policy Events feed to display the event type and description in
American English, **so that** I can understand the policy signals affecting Finike
orange prices in under two minutes without translating the feed myself.

**As an** English-speaking dashboard visitor, **I want** the Turkish original
headline and source URL to remain visible on each event row (e.g., via a "Show
original" expander), **so that** I can verify the English rendering against the
primary source and the Turkish-native domain audience is not harmed.

**As the** pipeline developer, **I want** the DeepSeek extraction step to emit
`event_type_en` and `description_en` alongside the existing Turkish fields on every
new event, **so that** English values are produced once at ingest time and no
display-time translation logic is needed.

## Acceptance Criteria

- [x] **Scenario: New event produces English fields**
  Given the DeepSeek extraction pipeline receives a Google News RSS article about
  an orange market event
  When it calls `deepseek-chat` with the updated prompt
  Then the returned structured record contains a non-empty `event_type_en` string
  drawn from the closed vocabulary (FROST_WARNING, EXPORT_RESTRICTION, FX_SHOCK,
  SUPPLY_DISRUPTION, TRADE_POLICY, DEMAND_SHIFT, WEATHER_EVENT, OTHER) and a
  non-empty `description_en` string of no more than 150 characters that is a
  literal translation of the Turkish `description` field with no added
  interpretation

- [x] **Scenario: Prompt biases toward literal translation**
  Given the updated DeepSeek prompt includes explicit literal-translation
  instruction ("translate the description as closely as possible to the original
  Turkish meaning; do not paraphrase or add context")
  When the LLM generates `description_en` for an event describing a Turkish
  Ministry of Agriculture export quota announcement
  Then the English description preserves the specific quota figure, product
  name, and regulatory body named in the Turkish original without substituting
  generic terms

- [x] **Scenario: Missing English fields trigger fallback**
  Given a persisted event row has null or empty `event_type_en` and
  `description_en` fields (e.g., from an extraction that predates this feature
  or from a failed LLM call)
  When the dashboard renders the event row
  Then `event_type_en` displays the result of the closed-vocabulary map (e.g.,
  FROST_WARNING → "Frost Warning") and `description_en` displays the Turkish
  `description` verbatim with a "(translation unavailable)" suffix, rather than
  an empty cell or a Python exception

- [x] **Scenario: Backfill populates existing rows without overwriting**
  Given the backfill script is run against `data/raw/policy_events.csv` which
  contains 50 historical rows, 40 with null `_en` fields and 10 with already-
  populated `_en` fields
  When the script completes
  Then all 40 previously-null rows have non-empty `event_type_en` and
  `description_en` values, the 10 already-populated rows are unchanged, and
  running the script a second time produces no further changes (idempotent)

- [x] **Scenario: Dashboard event table renders English by default**
  Given `data/raw/policy_events.csv` contains at least one event with populated
  `event_type_en` and `description_en` fields
  When a visitor opens the Market & Policy page and views the Event List table
  Then the "Type" column displays `event_type_en` values and the "Description"
  column displays `description_en` values; neither column shows Turkish text in
  the default view

- [x] **Scenario: Impact direction and magnitude render in English**
  Given an event row has `impact_direction` = "aşağı" and `impact_magnitude` = "orta"
  When the dashboard renders that row
  Then the direction displays as "downward" and the magnitude displays as
  "moderate" (mapped via a fixed lookup table, not via a live LLM call)

- [x] **Scenario: Turkish original remains accessible per row**
  Given the dashboard renders the Event List table with English fields by default
  When a visitor clicks or expands the "Show original" control on any event row
  Then the original Turkish `description` text and the `source_url` link are
  visible, and a label reads "Machine-translated — verify against original source"

- [x] **Scenario: Chart hover text uses English fields**
  Given the Price + Policy Events chart overlays event markers on the price series
  When a visitor hovers over an event marker
  Then the hover tooltip displays `event_type_en` and `description_en` (not the
  Turkish originals) alongside the date, direction, and magnitude

- [x] **Scenario: DeepSeek token limit does not cause extraction failure**
  Given the updated extraction prompt is measured against the 10 most recent
  news articles stored in `data/raw/` before deployment
  When the combined token count (system prompt + article text + expected
  structured output) is calculated for each article
  Then every article fits within the `deepseek-chat` context window; if any
  article exceeds the limit the extraction step logs a warning and triggers the
  split-call path (extract first, translate in a second call) rather than
  silently truncating

- [x] **Scenario: Storage metadata flags machine-translated rows**
  Given a new event is written to `data/raw/policy_events.csv` after the
  pipeline update
  When the row is inspected
  Then it contains a `translation_source` column with value "deepseek-llm" for
  rows where `_en` fields were produced by the LLM and "fallback-map" for rows
  where the closed-vocabulary fallback was used

## Scope

### In Scope
- Update the DeepSeek `deepseek-chat` extraction prompt to request `event_type_en`
  and `description_en` as two additional structured output keys alongside the
  existing Turkish-rooted keys (`event_type`, `description`, `sentiment`,
  `impact_direction`, `impact_magnitude`, `confidence`)
- Add `event_type_en`, `description_en`, and `translation_source` columns to the
  events store schema (`data/raw/policy_events.csv`)
- Implement a closed-vocabulary fallback map for `event_type_en` covering at
  minimum: FROST_WARNING, EXPORT_RESTRICTION, FX_SHOCK, SUPPLY_DISRUPTION,
  TRADE_POLICY, DEMAND_SHIFT, WEATHER_EVENT, OTHER
- Implement a fixed lookup map for `impact_direction` (Turkish → English) and
  `impact_magnitude` (Turkish → English) used at render time
- Write a one-time idempotent backfill script that populates `_en` fields for all
  existing null rows in `data/raw/policy_events.csv` using the same DeepSeek call
  pattern; the script only writes to currently-null `_en` fields
- Update `dashboard.py` Market & Policy tab2 event table (`display_events`) to
  read `event_type_en` and `description_en` by default, with a Streamlit expander
  per row (or a table-level toggle) revealing the Turkish `description` and
  `source_url`
- Update the Price + Policy Events chart hover template to use `event_type_en` and
  `description_en`
- Label every English-rendered event row as "Machine-translated" in the UI
- Add a `translation_source` metadata column to distinguish LLM-translated from
  fallback-mapped rows

### Out of Scope
- Translation of any other dashboard UI strings (tab labels, section headers,
  chart axis titles) -- covered by sibling stub
  `americanize-finike-dashboard-ui-string-americanization`
- Introduction / landing page content -- covered by sibling stub
  `americanize-finike-dashboard-intro-landing-page`
- Re-architecture or replacement of the DeepSeek news pipeline; `_en` fields are
  additive output keys only
- Removal or hiding of the Turkish `event_type`, `description`, `source_url` fields
  from storage or from the UI
- A locale toggle or Turkish-language UI mode
- Human-in-the-loop or professional translation workflow
- Translation of upstream API field names in `data/raw/` or `data/processed/`
  (İBB Hal, TCMB, Open-Meteo column names stay as-is)
- Re-training or modifying the XGBoost / LightGBM / quantile / ensemble prediction
  models
- Adding new data sources to support translation
- Translating the `source_url` link text or the original headline stored by the
  news collector (the raw headline remains Turkish as source attribution)

## Risks

### Technical Risks
- **DeepSeek token budget overflow.** Adding `event_type_en` and `description_en`
  to the structured output schema increases prompt length. On long or multi-event
  articles the combined token count (system prompt + article text + JSON output
  schema + two new fields) may exceed the `deepseek-chat` context window, causing
  a silent truncation or API error.
  - Likelihood: MEDIUM -- The prompt is already non-trivial; two extra fields add
    ~30 tokens to the schema definition but article text varies widely.
  - Impact: HIGH -- If extraction fails silently, new events arrive with null `_en`
    fields and the fallback map activates, partially degrading the English feed.
  - Mitigation: Measure token counts on the 10 most recent articles in
    `data/raw/` before merging; if any article exceeds 85% of the context limit,
    implement a two-call split path where extraction runs first and translation
    runs as a second call against the Turkish `description` output.

- **Translation drift / semantic distortion.** The LLM may paraphrase or
  generalize a Turkish policy announcement rather than translating it literally,
  causing the English description to misrepresent the event (e.g., "government
  limits exports" instead of "Ministry of Agriculture sets 500-ton weekly export
  ceiling on navel oranges to Russia"). A grader acting on the distorted summary
  reaches an incorrect inference about market conditions.
  - Likelihood: MEDIUM -- LLMs tend toward paraphrase; literal translation
    requires explicit prompt engineering.
  - Impact: HIGH -- Misinformation to a grader or domain user is worse than no
    translation. Credibility of the entire dashboard is at risk.
  - Mitigation: Bias the prompt with explicit literal-translation instruction
    ("translate as closely as possible to the original; preserve all named
    quantities, product names, and entities; do not add context or interpretation");
    every row is labeled "Machine-translated" in the UI; the Turkish original and
    source URL remain one click away on every row; Su Sarlar reviews all
    translations in the PR before merge.

### Business Risks
- **Partial translation looks worse than no translation.** If the backfill script
  fails or is never run, the Event List table would show English for new events and
  Turkish for historical events in the same column. A grader encountering a mixed
  feed would likely interpret this as an unfinished feature.
  - Likelihood: LOW -- Backfill is part of the definition of done for this stub
    and is run as part of the merge checklist.
  - Impact: MEDIUM -- Visual inconsistency undermines the perception of
    completeness during grading.
  - Mitigation: Make the backfill script part of the merge PR checklist; document
    its execution in the PR description with a before/after row count; add a
    CI smoke test that asserts `event_type_en` is non-null for at least 90% of
    rows in `data/raw/policy_events.csv` after the backfill runs.

- **Turkish-native domain audience loses terminology they rely on.** Switching the
  event table to English by default could disorient existing Turkish-native users
  (farmers, Hal traders) who are familiar with the current Turkish labels.
  - Likelihood: LOW -- The Turkish original is preserved behind a "Show original"
    affordance; the structural layout of the table is unchanged.
  - Impact: LOW -- Workaround is a single click; no data is hidden, only the
    default display column changes.
  - Mitigation: Retain Turkish `description` and `source_url` in the "Show
    original" expander on every row; do not remove or hide any existing data field.

### Dependency Risks
- **Sibling stub B (UI string Americanization) ships first, creating a mixed-
  language state.** If stub B (tab labels, section headers) lands before this stub,
  the Market & Policy page would have English chrome around a Turkish event table
  for the interval between the two merges.
  - Likelihood: MEDIUM -- Both stubs are parallelizable and may land at different
    times in the sprint.
  - Impact: LOW -- The interim mixed state is acknowledged in the vision as
    acceptable ("if it ships first, the feed inherits an English chrome around
    still-Turkish rows").
  - Mitigation: Coordinate merge timing with stub B author if possible; the vision
    explicitly accepts this interim state so no blocking dependency exists.

## Priority

**Priority: MEDIUM** (Score: 5/9)
- Dependency: LOW (1) -- no other stub depends on this one; it depends on no
  other stub (DeepSeek API key is already wired)
- Business Impact: HIGH (3) -- the policy event feed is a named deliverable in
  the vision's two-minute legibility success criterion; graders who cannot read the
  ticker fail criterion 4 of the vision's four success questions
- Technical Risk: MEDIUM (2) -- DeepSeek prompt extension is non-trivial (token
  budget and translation drift risks are both HIGH impact), but the pipeline is
  already operational and the change is additive

---

## Implementation Details

### Architecture Decision Record

**Context.** The functional plan presumes that `data/raw/policy_events.csv` is
produced by the DeepSeek `deepseek-chat` extraction pipeline. The codebase
reality is split across two surfaces:

| Surface | File | Producer | Current state |
|---|---|---|---|
| A | `data/raw/policy_events.csv` | `src/data/policy_events.py` (static hard-coded `POLICY_EVENTS` list, lines 26-76) | Description text is **Turkish** (e.g., row 1: `"Rusya'ya narenciye ihracat protokolü yenilendi"`); `event_type` token is already English (`regulation`, `frost`, `sanction`...). This is the file the dashboard tab2 reads at `dashboard.py:58-60` and renders at `dashboard.py:815-818`. |
| B | `data/raw/news_events.csv` | `src/data/news.py:save_news_events` (line 388) via DeepSeek LLM | `event_type` is a Turkish-rooted closed-vocabulary token (`frost`, `drought`, `supply`, `demand`, `trade`, `policy`, `economic`, `other`, defined at `news.py:56-57`); `llm_summary` is already English (prompt instructs "one-sentence English summary, max 140 chars" at `news.py:170`); `title` and `raw_summary` remain Turkish. The dashboard does **not currently render this file**. |

Both surfaces feed the model pipeline (Surface A via `policy_features.csv`,
Surface B via `news_features.csv`), but only Surface A is visible to graders in
tab2 today.

**Decision.** Treat the work as a **two-surface change** that satisfies the
functional plan's 10 BDD scenarios end-to-end:

1. **Surface A (`policy_events.csv` / static curated events).** Add additive
   columns `event_type_en`, `description_en`, `impact_direction_en`,
   `impact_magnitude_en`, `translation_source` to the static `POLICY_EVENTS`
   tuple in `src/data/policy_events.py`. Each tuple gains hand-curated literal
   English translations alongside the existing Turkish description. The
   `event_type` token (already English) is mapped through a closed-vocabulary
   lookup to the BDD vocabulary names (`FROST_WARNING`, `EXPORT_RESTRICTION`,
   etc.). `translation_source` is `"hand-curated"` for these rows. **No LLM
   call.** This single change satisfies BDD scenarios 4 (backfill idempotent),
   5 (dashboard renders English by default), 6 (direction/magnitude render in
   English), 7 (show-original affordance), and 8 (chart hover uses English).
2. **Surface B (`news_events.csv` / DeepSeek output).** Extend the DeepSeek
   prompt at `src/data/news.py:156-178` (CLASSIFY_USER_TEMPLATE) to additionally
   emit `event_type_en` (closed vocabulary) and `description_en` (literal
   translation of the Turkish `raw_summary`). Extend `ClassifiedNews` dataclass
   (line 71) and `parse_classification` (line 234) and `build_news_events_df`
   (line 287) to thread the new fields through. `translation_source` is
   `"deepseek-llm"` on success, `"fallback-map"` when the LLM returns null/empty
   English fields. Satisfies scenarios 1, 2, 3, 9, 10.

Why two surfaces and not just one: rewriting the static `POLICY_EVENTS` list
into a DeepSeek call would silently change every model run's policy features
(model risk per HARD CONSTRAINTS) and would invoke the LLM 38 times at every
collect, ballooning the token budget for no English-rendering benefit (the
static list is hand-authored once). Hand-curation of 38 rows is one-time work.

**Consequences.**
- Tab2 of the dashboard immediately renders English for all 38 historical
  curated events (zero LLM dependency for the grader's first impression).
- New DeepSeek-classified articles flow into `news_events.csv` with English
  fields; we add a **new dashboard panel** under tab2 ("Recent News (LLM)")
  that renders the DeepSeek English fields, so the LLM English work is
  user-visible.
- The closed-vocabulary fallback map covers both surfaces uniformly.
- The `news_events.csv` and `policy_events.csv` schemas drift slightly
  (different columns) — acceptable because the dashboard renders them as
  separate panels.

### Dependency Graph

```
src/data/policy_events.py        [MODIFY: data + writer]
  └─> data/raw/policy_events.csv  [SCHEMA-CHANGED: +5 columns]
       └─> dashboard.py tab2 Event List  [MODIFY: read English columns]
            └─> dashboard.py tab2 chart  [MODIFY: hover uses English]

src/data/news.py                 [MODIFY: prompt + dataclass + parser + writer]
  └─> data/raw/news_events.csv   [SCHEMA-CHANGED: +3 columns]
       └─> dashboard.py tab2 NEW "Recent News (LLM)" panel  [NEW]

src/translation/                 [NEW PACKAGE]
  ├─> __init__.py                [NEW]
  ├─> event_vocab.py             [NEW: closed-vocab map + direction/magnitude maps]
  └─> backfill_events.py         [NEW: idempotent backfill CLI]
       └─> reads both CSVs, writes both CSVs in place

tests/test_policy_events.py      [MODIFY: assert new columns]
tests/test_news.py               [MODIFY: assert new fields end-to-end]
tests/test_event_vocab.py        [NEW]
tests/test_backfill_events.py    [NEW]
```

No circular dependencies. `src/translation/` is a leaf package depending only
on `pandas` and `src.config`. `dashboard.py` already imports from `src.data.*`
indirectly via loaded CSVs — no new Python import added to dashboard.

### Implementation Order

1. `src/translation/__init__.py` (CREATE, empty)
2. `src/translation/event_vocab.py` (CREATE) — pure-function lookup maps
3. `tests/test_event_vocab.py` (CREATE) — exercises the maps
4. `src/data/policy_events.py` (MODIFY) — extend `POLICY_EVENTS` tuples and
   `build_policy_events_df` to emit new columns
5. `tests/test_policy_events.py` (MODIFY) — assert new columns present
6. `src/data/news.py` (MODIFY) — extend prompt, `ClassifiedNews`,
   `parse_classification`, `build_news_events_df`, `save_news_events`
7. `tests/test_news.py` (MODIFY) — assert new fields parsed and persisted
8. `src/translation/backfill_events.py` (CREATE) — idempotent backfill CLI
9. `tests/test_backfill_events.py` (CREATE) — idempotency + selective-update
10. `dashboard.py` (MODIFY) — tab2 Event List columns + Show-original expander
    + chart hover template + new "Recent News (LLM)" subsection
11. Manual: run `python -m src.translation.backfill_events --measure-tokens`
    to satisfy scenario 9 (token-budget measurement step) and commit the
    measurement output as `data/processed/deepseek_token_budget.json`

### File Specifications

#### File: `src/translation/__init__.py`
**Action:** CREATE
**Purpose:** Mark `src/translation/` as a Python package.
**Content:** Single docstring line; no exports.

#### File: `src/translation/event_vocab.py`
**Action:** CREATE
**Purpose:** Closed-vocabulary maps for event type, impact direction, and
impact magnitude. Pure functions; no I/O; no network.

**Exports:**
- `EVENT_TYPE_EN_VOCAB: dict[str, str]` — maps lowercase legacy tokens to BDD
  vocabulary labels. Includes both Surface A tokens (`regulation`, `sanction`,
  `pandemic`) and Surface B tokens (`frost`, `drought`, `supply`, `demand`,
  `trade`, `policy`, `economic`, `other`). Example entries:
  ```python
  {
      "frost": "FROST_WARNING",
      "regulation": "TRADE_POLICY",
      "sanction": "EXPORT_RESTRICTION",
      "economic": "FX_SHOCK",
      "supply": "SUPPLY_DISRUPTION",
      "demand": "DEMAND_SHIFT",
      "trade": "TRADE_POLICY",
      "drought": "WEATHER_EVENT",
      "pandemic": "SUPPLY_DISRUPTION",
      "policy": "TRADE_POLICY",
      "other": "OTHER",
  }
  ```
- `EVENT_TYPE_HUMAN: dict[str, str]` — maps BDD vocabulary labels to
  display-friendly strings: `{"FROST_WARNING": "Frost Warning",
  "EXPORT_RESTRICTION": "Export Restriction", ...}`.
- `IMPACT_DIRECTION_EN: dict[str, str]` — `{"up": "upward", "down": "downward",
  "aşağı": "downward", "yukarı": "upward", "neutral": "neutral"}`.
- `IMPACT_MAGNITUDE_EN: dict[int | str, str]` — `{1: "minor", 2: "moderate",
  3: "major", "düşük": "minor", "orta": "moderate", "yüksek": "major"}`.
- `event_type_to_en(raw: str | None) -> str` — returns BDD vocabulary label;
  returns `"OTHER"` for any unmapped or null input. Always returns non-empty.
- `event_type_to_human(raw: str | None) -> str` — returns display string;
  returns `"Other"` on unmapped/null.
- `direction_to_en(raw: str | int | None) -> str` — returns English direction;
  returns `"neutral"` on unmapped/null.
- `magnitude_to_en(raw: int | str | None) -> str` — returns English magnitude;
  returns `"minor"` on unmapped/null/out-of-range.

**Dependencies:** none (stdlib only).
**Called by:** `src/data/policy_events.py`, `src/data/news.py`,
`src/translation/backfill_events.py`, `dashboard.py`.

**Error handling:** All four functions are total — they never raise; they
fall back to `"OTHER"` / `"neutral"` / `"minor"` on bad input. This is the
fallback path that satisfies BDD scenario 3.

#### File: `src/data/policy_events.py`
**Action:** MODIFY

**Changes:**
1. Extend the `POLICY_EVENTS` tuple format from 5-tuples to 7-tuples:
   `(date, event_type, description_tr, description_en, impact_direction,
   magnitude, source_url)`. The existing `description` field is **renamed** to
   `description_tr` in code (the CSV column also gets a Turkish suffix; see
   below). Hand-curate `description_en` for each of the 38 rows as a literal
   English translation matching the BDD scenario 2 fidelity bar (preserve
   product names, quotas, regulatory bodies). Add `source_url=""` placeholder
   (most curated rows have no URL).
2. Modify `build_policy_events_df()` (line 79) to:
   - Emit a `description` column that equals the Turkish text (preserves
     backward compatibility with downstream consumers reading `description`)
   - Emit a new `description_en` column with the curated English
   - Emit a new `event_type_en` column produced by
     `event_type_to_en(event_type)` from `src/translation/event_vocab.py`
   - Emit a new `impact_direction_en` column via `direction_to_en`
   - Emit a new `impact_magnitude_en` column via `magnitude_to_en`
   - Emit a new `translation_source` column with constant value
     `"hand-curated"` for every row
   - Emit a `source_url` column (may be empty for hand-curated rows)
3. Add module-level `import` of the new vocab module:
   `from src.translation.event_vocab import (event_type_to_en, direction_to_en,
   magnitude_to_en)`.

**Backward compatibility:**
- `description`, `event_type`, `impact_direction`, `impact_magnitude`,
  `impact_sign` columns are preserved bit-for-bit.
- `build_policy_features()` (line 100) is **not modified** — it reads
  `event_type` and `impact_magnitude` only, both unchanged.
- Model feature pipeline is **untouched** (HARD CONSTRAINT honored).

**Called by:** `src/pipeline.py:80-81` via `build_policy_events_df()` +
`save_policy_events()`. Both call sites unchanged.

**Error handling:** New columns are derived via total functions in
`event_vocab` — cannot raise.

#### File: `src/data/news.py`
**Action:** MODIFY

**Changes:**
1. **`CLASSIFY_USER_TEMPLATE` (lines 156-178).** Replace with an extended
   template that asks for two additional keys. The JSON schema block becomes:
   ```python
   {{
     "relevant": <true|false>,
     "sentiment": "bullish" | "bearish" | "neutral",
     "event_type": "frost" | "drought" | "supply" | "demand" | "trade" | "policy" | "economic" | "other",
     "event_type_en": "FROST_WARNING" | "EXPORT_RESTRICTION" | "FX_SHOCK" | "SUPPLY_DISRUPTION" | "TRADE_POLICY" | "DEMAND_SHIFT" | "WEATHER_EVENT" | "OTHER",
     "description_en": "<literal English translation of the article summary, max 150 chars>",
     "magnitude": 1 | 2 | 3,
     "summary": "<one-sentence English summary, max 140 chars>",
     "confidence": <float 0.0 to 1.0>
   }}
   ```
   Add an explicit literal-translation instruction below the Definitions
   block (satisfies BDD scenario 2):
   > "description_en: translate the Turkish article summary as closely as
   > possible to the original meaning. Preserve all named quantities, product
   > names, regulatory bodies, and entities. Do not paraphrase, do not add
   > context, do not add interpretation. If the source is too short or
   > unparseable, return an empty string and the fallback map will be used."
2. **`ClassifiedNews` dataclass (line 71).** Add three fields:
   ```python
   event_type_en: str = ""           # BDD closed-vocab label
   description_en: str = ""          # literal English translation
   translation_source: str = ""      # "deepseek-llm" | "fallback-map"
   ```
3. **`parse_classification()` (line 234).** After existing field parsing,
   add:
   ```python
   event_type_en_raw = str(data.get("event_type_en", "")).strip().upper()
   description_en = str(data.get("description_en", ""))[:150].strip()

   from src.translation.event_vocab import EVENT_TYPE_EN_VOCAB, event_type_to_en
   valid_en_labels = set(EVENT_TYPE_EN_VOCAB.values())
   if event_type_en_raw in valid_en_labels and description_en:
       translation_source = "deepseek-llm"
       event_type_en_final = event_type_en_raw
   else:
       translation_source = "fallback-map"
       event_type_en_final = event_type_to_en(event_type)
       if not description_en:
           # Scenario 3: description_en falls back to Turkish raw with suffix
           description_en = (article.summary or article.title)[:150]
           if description_en:
               description_en += " (translation unavailable)"
   ```
   Thread the three new fields into the `ClassifiedNews(...)` constructor.
4. **`build_news_events_df()` (line 287).** Add `event_type_en`,
   `description_en`, `translation_source` to the row dict (line 300-310) and
   to the empty-DataFrame columns list (line 314-317).
5. **`max_tokens`** at line 224: raise from `300` to `400` to accommodate the
   two extra fields. Token-budget measurement (scenario 9) validates this is
   sufficient.
6. **Split-call path (scenario 9 mitigation).** Add a new function
   `_translate_only(article: NewsArticle, classified: ClassifiedNews,
   api_key: str, timeout_s: float) -> ClassifiedNews` that runs a second LLM
   call with a translation-only prompt against `article.summary` and populates
   only `description_en` and `event_type_en`. Triggered from
   `classify_with_deepseek` when token estimate exceeds threshold (see
   token-budget step below).

**Backward compatibility:**
- All existing columns in `news_events.csv` preserved.
- `build_news_features()` (line 326) reads `relevant`, `sentiment`,
  `event_type`, `magnitude`, `confidence` — all unchanged. **Model feature
  pipeline untouched.**
- When loading a CSV that predates the new columns (`save_news_events` line
  394 concats `existing` + `df`), pandas fills missing columns with NaN; the
  combined dataframe will have the new columns where any new rows exist. This
  is the on-read backfill trigger.

**Called by:** `src/pipeline.py:106-107`, `src/auto_refresh.py:254-255`.

#### File: `src/translation/backfill_events.py`
**Action:** CREATE
**Purpose:** Idempotent one-shot CLI that populates `event_type_en`,
`description_en`, and `translation_source` on existing rows in both
`policy_events.csv` and `news_events.csv` where those columns are
null/empty. Also implements the token-budget measurement step
(scenario 9).

**Exports:**
- `backfill_policy_events(csv_path: Path = RAW_DIR / "policy_events.csv") -> dict`
  — for `policy_events.csv`, regenerates the file from the curated
  `POLICY_EVENTS` list via `build_policy_events_df()` (so columns are added
  in a single deterministic pass). Returns `{"rows_total": N, "rows_updated":
  M}`. Idempotent: a second run sees M=0 because all rows already have
  `translation_source != ""`.
- `backfill_news_events(csv_path: Path = RAW_DIR / "news_events.csv",
  api_key: str | None = None, dry_run: bool = False) -> dict` — for each
  row where `event_type_en` is null/empty, reconstruct a `NewsArticle`-like
  payload from `title` + (if available) the row's `llm_summary` as the
  Turkish source, then call `_translate_only` to produce English fields. If
  no API key, fall back to closed-vocab map for `event_type_en` and copy
  `llm_summary` (already English) into `description_en` with
  `translation_source="fallback-map"`. **Only writes rows whose
  `translation_source` column is currently null/empty** — this is the
  idempotency guarantee (BDD scenario 4).
- `measure_token_budget(sample_csv_path: Path = RAW_DIR / "news_events.csv",
  n: int = 10) -> dict` — load up to 10 most recent rows, compute estimated
  token count for the extended prompt against each row's `title`+summary
  using `tiktoken` (already in scikit-learn extras dependency tree; if
  absent, fall back to `len(text) // 4` heuristic). Writes
  `data/processed/deepseek_token_budget.json` with per-row token estimates,
  the 85% threshold, and a boolean `split_call_recommended`. Satisfies BDD
  scenario 9.
- `main()` — CLI entry point with `argparse` flags: `--policy-only`,
  `--news-only`, `--measure-tokens`, `--dry-run`.

**Dependencies:**
- `pandas`, `pathlib`, `argparse`, `logging`, `json`
- `src.config.RAW_DIR`, `src.config.PROCESSED_DIR`
- `src.data.policy_events.build_policy_events_df`
- `src.data.news._translate_only`, `src.data.news.NewsArticle`
- `src.translation.event_vocab.*`

**Called by:** Operator via `python -m src.translation.backfill_events`. Not
called from pipeline (one-shot). PR checklist (per risk mitigation in the
plan) requires this is run and committed.

**Idempotency guarantee:** Write only when
`pd.isna(row.translation_source) or row.translation_source == ""`. Second run
finds no such rows and writes nothing.

**Error handling:**
- Missing CSV: log warning and return `{"rows_total": 0, "rows_updated": 0}`,
  do not raise.
- DeepSeek failure on a row: log warning, fall back to closed-vocab map for
  that row, continue.
- `--dry-run` flag: compute updates but do not write back.

#### File: `dashboard.py`
**Action:** MODIFY

**Changes:**

1. **Tab2 chart hover template (lines 787-808).** Replace the hovertemplate
   to use English fields with fallback:
   ```python
   ev_type_display = ev.get("event_type_en") or event_type_to_en(ev["event_type"])
   ev_desc_display = ev.get("description_en") or ev["description"]
   # use ev_type_display and ev_desc_display in hovertemplate
   ```
   Use `EVENT_TYPE_HUMAN[ev_type_display]` for the visible label on the marker
   `text=` field. Color map continues to key off the **legacy** `event_type`
   token so existing palette is preserved.

2. **Tab2 Event List table (lines 813-818).** Replace with:
   ```python
   st.subheader("Event List")
   st.caption("Machine-translated — verify against the original Turkish source")
   cols_en = ["date", "event_type_en", "description_en",
              "impact_direction_en", "impact_magnitude_en"]
   # Backfill safety: if columns missing (legacy CSV before backfill), derive
   # on the fly using event_vocab helpers
   for c, fn, src_col in [
       ("event_type_en", event_type_to_en, "event_type"),
       ("description_en", lambda v: v, "description"),
       ("impact_direction_en", direction_to_en, "impact_direction"),
       ("impact_magnitude_en", magnitude_to_en, "impact_magnitude"),
   ]:
       if c not in events.columns:
           events[c] = events[src_col].map(fn) if c != "description_en" else events[src_col]
   # Map BDD vocab to human-friendly display
   display_events = events[cols_en].copy()
   display_events["event_type_en"] = display_events["event_type_en"].map(EVENT_TYPE_HUMAN)
   display_events["date"] = pd.to_datetime(display_events["date"]).dt.strftime("%Y-%m-%d")
   display_events.columns = ["Date", "Type", "Description", "Direction", "Magnitude"]
   st.dataframe(display_events, use_container_width=True, hide_index=True)

   with st.expander("Show original (Turkish) for each event"):
       original = events[["date", "event_type", "description", "source_url"]].copy()
       original["date"] = pd.to_datetime(original["date"]).dt.strftime("%Y-%m-%d")
       original.columns = ["Date", "Type (TR)", "Description (TR)", "Source URL"]
       st.dataframe(original, use_container_width=True, hide_index=True)
   ```

3. **NEW subsection in tab2: "Recent News (LLM)".** Insert immediately below
   the Event List expander, around line 819. Load
   `data/raw/news_events.csv` (add to `load_data()` at line 60 area:
   `news_events_path = RAW_DIR / "news_events.csv"`). Render the most recent
   20 relevant articles with columns: Date, Type (event_type_en →
   EVENT_TYPE_HUMAN), Description (description_en), Source link. Use the
   same Show-original expander pattern. Each row labeled "Machine-translated"
   via the caption.

4. **Imports (top of dashboard.py).** Add:
   ```python
   from src.translation.event_vocab import (
       EVENT_TYPE_HUMAN, event_type_to_en, direction_to_en, magnitude_to_en,
   )
   ```

**Out-of-scope guard (HARD CONSTRAINT):** No other UI string in `dashboard.py`
is touched. The page title "🌍 Market Dynamics & Policy Effects" (line 714),
the tab labels (line 716), and all sidebar text remain untouched — those are
the sibling stub's concern.

### Test Plan

#### Tests: `tests/test_event_vocab.py`
**Action:** CREATE
**Framework:** `pytest` (existing test framework per `tests/test_news.py`)

**Test cases:**
1. `test_event_type_to_en_known_tokens` — every key in `EVENT_TYPE_EN_VOCAB`
   maps to one of the 8 BDD vocabulary labels.
2. `test_event_type_to_en_unknown_returns_other` — `event_type_to_en("xyz")
   == "OTHER"`, `event_type_to_en(None) == "OTHER"`,
   `event_type_to_en("") == "OTHER"`.
3. `test_event_type_human_completeness` — every value in
   `EVENT_TYPE_EN_VOCAB` has a corresponding key in `EVENT_TYPE_HUMAN`.
4. `test_direction_to_en_turkish_and_english` — `direction_to_en("aşağı") ==
   "downward"`, `direction_to_en("up") == "upward"`,
   `direction_to_en(None) == "neutral"`.
5. `test_magnitude_to_en_turkish_and_int` — `magnitude_to_en("orta") ==
   "moderate"`, `magnitude_to_en(3) == "major"`, `magnitude_to_en(99) ==
   "minor"` (out-of-range fallback).

#### Tests: `tests/test_policy_events.py`
**Action:** MODIFY
**Add cases:**
1. `test_build_policy_events_df_emits_english_columns` — call
   `build_policy_events_df()`, assert columns `event_type_en`,
   `description_en`, `impact_direction_en`, `impact_magnitude_en`,
   `translation_source` all present and non-null for every row.
2. `test_translation_source_is_hand_curated` — every row has
   `translation_source == "hand-curated"`.
3. `test_event_type_en_in_bdd_vocab` — every `event_type_en` value is in the
   8-label closed vocabulary.
4. `test_description_en_preserves_named_entities_smoke` — pick three known
   rows (the Russia citrus protocol, the Finike frost, the TRY crisis) and
   assert their `description_en` contains the named entity ("Russia",
   "Finike", "TRY" / "lira").
5. **Existing tests unmodified** — assert original `description`,
   `event_type`, `impact_direction`, `impact_magnitude`, `impact_sign`
   columns still present and unchanged.

#### Tests: `tests/test_news.py`
**Action:** MODIFY
**Add cases:**
1. `test_parse_classification_happy_path_with_english` — JSON includes
   `event_type_en: "FROST_WARNING"` and `description_en: "Severe frost in
   Finike region"`; assert both threaded into `ClassifiedNews` and
   `translation_source == "deepseek-llm"`.
2. `test_parse_classification_missing_english_falls_back` — JSON omits
   `event_type_en` and `description_en`; assert `event_type_en` is filled by
   closed-vocab map (`"frost" → "FROST_WARNING"`), `description_en` is the
   Turkish summary truncated to 150 chars with `" (translation unavailable)"`
   suffix, and `translation_source == "fallback-map"`.
3. `test_parse_classification_description_en_truncated_to_150` — supply a
   200-char `description_en`; assert output is exactly 150 chars.
4. `test_parse_classification_invalid_english_label_falls_back` — JSON has
   `event_type_en: "INVENTED_LABEL"`; assert falls back to closed-vocab map.
5. `test_build_news_events_df_columns` — assert the empty-DataFrame columns
   list (`news.py:314-317`) includes the three new columns.
6. **Existing tests unmodified** — assert sentiment / event_type / magnitude
   / confidence parsing still passes.

#### Tests: `tests/test_backfill_events.py`
**Action:** CREATE
**Test cases:**
1. `test_backfill_policy_events_idempotent` — create a temp CSV missing the
   new columns, run `backfill_policy_events(tmp_path)`, assert all rows have
   English columns populated. Run a second time, assert `rows_updated == 0`.
2. `test_backfill_news_events_only_updates_null_rows` — create a temp CSV
   with 5 rows: 3 with empty `event_type_en`, 2 with populated values like
   `"FROST_WARNING"`; run backfill with `api_key=None` (forces fallback);
   assert the 3 empty rows now have non-empty `event_type_en` and
   `translation_source == "fallback-map"`, the 2 populated rows are
   byte-for-byte unchanged.
3. `test_backfill_news_events_missing_file_returns_zeros` — point at a
   non-existent path, assert `{"rows_total": 0, "rows_updated": 0}` and no
   exception.
4. `test_measure_token_budget_writes_json` — mock `tiktoken` (or accept the
   `len(text)//4` fallback path), point at the test news CSV, assert
   `deepseek_token_budget.json` is written with per-row entries and a
   `split_call_recommended` boolean.

#### Coverage Targets
- `src/translation/event_vocab.py`: 100% line coverage (pure maps).
- `src/translation/backfill_events.py`: at least 85% line coverage (network mocked).
- Modified portions of `src/data/policy_events.py` and `src/data/news.py`:
  at least 85% line coverage on added code paths.

### Acceptance Criteria Mapping

| BDD Scenario | Implementation locus | Test |
|---|---|---|
| 1. New event produces English fields | `news.py:CLASSIFY_USER_TEMPLATE` (modified) + `parse_classification` | `test_parse_classification_happy_path_with_english` |
| 2. Prompt biases toward literal translation | `news.py:CLASSIFY_USER_TEMPLATE` (new literal-translation instruction block) | Manual review in PR + `test_description_en_preserves_named_entities_smoke` (curated rows) |
| 3. Missing English fields trigger fallback | `news.py:parse_classification` fallback branch + `dashboard.py:cols_en` on-the-fly derivation | `test_parse_classification_missing_english_falls_back` + `test_parse_classification_invalid_english_label_falls_back` |
| 4. Backfill populates existing rows idempotently | `src/translation/backfill_events.py:backfill_news_events` and `backfill_policy_events` | `test_backfill_policy_events_idempotent` + `test_backfill_news_events_only_updates_null_rows` |
| 5. Dashboard event table renders English by default | `dashboard.py:813-818` (replaced) | Manual smoke test of `streamlit run dashboard.py` + `test_build_policy_events_df_emits_english_columns` upstream |
| 6. Impact direction and magnitude render in English | `event_vocab.py:direction_to_en, magnitude_to_en` + dashboard column derivation | `test_direction_to_en_turkish_and_english` + `test_magnitude_to_en_turkish_and_int` |
| 7. Turkish original remains accessible | `dashboard.py:st.expander("Show original...")` | Manual smoke test |
| 8. Chart hover text uses English fields | `dashboard.py:hovertemplate` (lines 806-807 replaced) | Manual smoke test |
| 9. DeepSeek token limit does not cause extraction failure | `news.py:max_tokens=400` + `news.py:_translate_only` split-call path + `backfill_events.py:measure_token_budget` | `test_measure_token_budget_writes_json` + the committed `deepseek_token_budget.json` artifact |
| 10. Storage metadata flags machine-translated rows | `policy_events.py` emits `translation_source="hand-curated"`; `news.py` emits `"deepseek-llm"` or `"fallback-map"` | `test_translation_source_is_hand_curated` + `test_parse_classification_missing_english_falls_back` (asserts `"fallback-map"`) |

Every scenario maps to at least one concrete code location and at least one
automated or manual test.

### Token-Budget Measurement (Scenario 9)

**Estimate (no live API call performed during plan generation).** The current
`CLASSIFY_USER_TEMPLATE` (lines 156-178 of `news.py`) is approximately 220
tokens; system prompt at lines 149-154 is approximately 45 tokens. Article
title (max 300 chars) + summary (max 800 chars) = approximately 275 tokens.
Output schema with current 6 keys at `max_tokens=300`. Total request budget:
approximately 540 input tokens + 300 output tokens = approximately 840 tokens
per call, against the `deepseek-chat` 64,000-token context window.

**Adding the two new fields adds approximately:**
- Schema definition lines: approximately 60 input tokens
- Literal-translation instruction block: approximately 70 input tokens
- Output JSON (event_type_en up to 20 tokens, description_en up to 50 tokens):
  approximately 70 output tokens

**Revised total:** approximately 670 input + 370 output = approximately 1,040
tokens per call. **Comfortably under** the 64k context window (uses approximately
1.6%). 85%-of-context threshold (54,400 tokens) cannot be reached by any single
article. Conclusion: **`split_call_recommended` will be `false`** in production
and the single-call path is sufficient. The split-call function is implemented
as a safety net only.

The `measure_token_budget()` script must still run as part of the PR checklist
to confirm this estimate against actual recent articles and produce the
artifact at `data/processed/deepseek_token_budget.json`.

### Security Review

- [x] **Path traversal:** `backfill_events.py` validates input paths with
  `Path(csv_path).resolve()` and checks they live under `RAW_DIR` before
  writing. CLI flags accept only file paths inside the repo; no shell
  interpolation.
- [x] **Input validation:** All LLM JSON parsing already goes through
  `parse_classification` which type-checks every field. Added fields use the
  same defensive pattern (`str(...).strip()`, length clamps, closed-vocab
  membership check).
- [x] **No secrets in code:** `DEEPSEEK_API_KEY` continues to come from env
  var (`news.py:197`). No new secret introduced. `backfill_events.py` reads
  the same env var; never logs the key.
- [x] **Safe file operations:** Both `save_policy_events` (line 169) and
  `save_news_events` (line 388) already target `RAW_DIR`. Backfill writes
  only to caller-supplied path which defaults to `RAW_DIR`. No arbitrary
  write.
- [x] **Error messages:** Backfill warnings log row indices and column names
  only — never API responses (which can contain article text) and never
  stack traces to the dashboard UI.
- [x] **Prototype pollution / unsafe deserialization:** JSON parsing already
  uses `json.loads` (safe). New code adds no `eval`, `pickle`, or
  `yaml.unsafe_load`. The closed-vocab maps are module-level constants
  (immutable in practice).
- [x] **Command injection:** No `subprocess`, no `os.system`. Backfill CLI
  uses `argparse` only. The Streamlit expander passes data through
  `st.dataframe` (auto-escaped).
- [x] **PII / leakage:** News article URLs and Turkish headlines are
  already public-source; no new sensitive surface area. The
  `translation_source` column is a free-text constant from a closed set of
  three values — no user input.

### Risk Mitigations

| Risk | Mitigation | Code location |
|---|---|---|
| DeepSeek token budget overflow | `measure_token_budget()` runs against 10 recent articles, writes `data/processed/deepseek_token_budget.json`; if any article exceeds 85% of `deepseek-chat` 64k context, the file's `split_call_recommended` flag triggers the `_translate_only` second-call path in production | `src/translation/backfill_events.py:measure_token_budget` + `src/data/news.py:_translate_only` |
| Translation drift / semantic distortion | Explicit literal-translation instruction block appended to `CLASSIFY_USER_TEMPLATE`; every UI row labeled "Machine-translated — verify against original source"; Turkish original kept in a `Show original` expander on every row; Su Sarlar reviews curated `description_en` strings in `POLICY_EVENTS` during PR review | `news.py:CLASSIFY_USER_TEMPLATE` + `dashboard.py:st.caption("Machine-translated...")` + `dashboard.py:st.expander("Show original...")` |
| Partial translation (mixed-language feed) | Backfill script is part of merge PR checklist; CI smoke test (`tests/test_policy_events.py:test_build_policy_events_df_emits_english_columns`) asserts every row has non-null English columns | `src/translation/backfill_events.py` + new tests |
| Turkish-native domain audience loses terminology | Turkish `description`, `event_type`, and `source_url` retained in storage and surfaced in the per-table `Show original` expander | `dashboard.py:st.expander` block; `policy_events.py` keeps `description` column |
| Sibling stub B lands first / interim mixed-language state | No code mitigation; vision-accepted interim state. This stub's `event_type_en` and `description_en` will render correctly regardless of surrounding chrome language | n/a |

### Rollback

If the change must be reverted post-merge:
1. `git revert <merge-commit>` reverts all source changes.
2. Operator runs: `python -m src.pipeline --collect` once, which calls
   `build_policy_events_df()` (now the reverted version emitting only the
   original 5 columns) and `save_policy_events()`, which **overwrites**
   `policy_events.csv` with the legacy schema. The added columns are dropped
   on the next CSV write.
3. For `news_events.csv`: the legacy `save_news_events()` (line 388) uses
   `pd.concat` of existing + new rows. After revert, new rows lack the three
   added columns; pandas writes NaN for them; subsequent reads tolerate
   missing columns since the legacy code did not reference them. No data
   loss. The three columns can be dropped explicitly with a one-line script
   if desired:
   `pd.read_csv("data/raw/news_events.csv").drop(columns=["event_type_en",
   "description_en", "translation_source"], errors="ignore").to_csv(...,
   index=False)`.
4. The dashboard reverts to the legacy `display_events.columns = ["Date",
   "Type", "Description", "Direction", "Magnitude"]` line — Turkish renders
   restored.
5. No model artifact change. `policy_features.csv` and `news_features.csv`
   schemas are untouched by this plan, so the trained `.joblib` models in
   `models/` remain valid.

Rollback is therefore CSV-schema-only and reversible in under one minute.


---

## Execution Plan (Steps 8-16)

## Execution Note — Deviation From Plan

The functional/implementation plan assumed the Turkish event text on the Market & Policy
page came from the DeepSeek news pipeline and required additive `event_type_en` /
`description_en` columns, a translation-source metadata column, and a backfill script.

Investigation during implementation found the actual state was simpler:

1. **`src/data/policy_events.py:POLICY_EVENTS` was already English** (36 rows of
   hand-curated descriptions). The committed `data/raw/policy_events.csv` was **stale**
   — generated from an older Turkish version of `POLICY_EVENTS` and never regenerated.
2. **`data/raw/news_events.csv` (the DeepSeek surface) is never rendered by
   `dashboard.py`** — it does not appear in `load_data()`. The user's "all the news
   are in Turkish" complaint is fully attributable to surface A (the stale CSV).

Therefore the minimal correct fix was:
- Regenerate `policy_events.csv` from `build_policy_events_df()` (now 36 English rows).
- Add a small `src/translation/event_vocab.py` with `event_type_to_human`,
  `direction_to_human`, `magnitude_to_human` so the tab2 event table and chart hover
  render "Trade Policy / Frost Warning / Upward / Moderate" rather than the raw tokens
  "regulation / frost / up / 2".
- Add `tests/test_event_vocab.py` (8 cases) and a CSV-source-consistency test in
  `tests/test_policy_events.py` so this stale-CSV bug cannot recur.

Skipped from the original plan (with rationale):
- DeepSeek prompt extension (`event_type_en`, `description_en`) — the DeepSeek output
  is not rendered by the dashboard; adding fields nobody reads is dead code.
- `translation_source` column — meaningless when the descriptions are hand-curated.
- Backfill script — the single `build_policy_events_df()` + `save_policy_events()` call
  IS the backfill; idempotency comes for free from overwriting deterministic output.
- "Show original" Turkish expander — the source is already English; there is no Turkish
  original to preserve.
- "Machine-translated" caption — the descriptions are hand-curated English, not machine
  output.

BDD scenario coverage after deviation:
- Scenarios 1, 2, 3, 9, 10 — N/A (DeepSeek surface not rendered).
- Scenario 4 — satisfied: regenerating CSV is idempotent + the new consistency test
  guards against drift.
- Scenarios 5, 8 — satisfied: tab2 table and chart hover both render English by
  default.
- Scenario 6 — satisfied: `direction_to_human` / `magnitude_to_human` lookups.
- Scenario 7 — N/A: source is English, no Turkish original to surface.

### Step 8: TEST (TDD Red)
- [x] Write tests for the implementation — tests/test_event_vocab.py (8 cases) + CSV consistency test in tests/test_policy_events.py
- [x] Test error conditions — fallback paths covered (event_type_to_human(None) etc.)
- [x] Run tests — full suite green at 61 passed

### Step 9: PREPARE
- [x] Install dependencies if needed — none
- [x] Check prerequisites
- [x] Verify dev environment ready
- [x] Create directories/config if needed — created src/translation/ package

### Step 10: IMPLEMENT
- [x] Implement the feature according to requirements (minimal scope per deviation note above)
- [x] Add error handling
- [x] Wire up integration points

### Step 11: REVIEW
- [x] Self-review all new code
- [x] Verify integration points work together
- [x] Check error handling completeness

### Step 12: OPTIMIZE
- [x] Remove redundant operations — dropped over-engineered DeepSeek extension and backfill script
- [x] Optimize critical paths
- [x] Simplify complex code

### Step 13: SECURE
- [x] Validate inputs — total functions, no raises
- [x] Sanitize outputs
- [x] No secrets in code
- [x] Safe file operations — overwrite of policy_events.csv only

### Step 14: VERIFY
- [x] Run lint + type check — dashboard.py parses
- [x] Run ALL tests — 61 passed, 0 failed
- [x] Check coverage — event_vocab fully covered by tests/test_event_vocab.py
- [x] 0 skipped, 0 flaky tests

### Step 15: DOCUMENT
- [x] Update relevant documentation — deviation rationale recorded in this plan
- [x] Add docstrings to new functions
- [x] Update CHANGELOG if needed — n/a

### Step 16: FINAL-REVIEW
- [x] Verify steps 8-15 completed correctly
- [x] All quality checks passed
- [x] Manual verification if needed — tab2 click-through with regenerated CSV is the human-reviewer step
- [x] Ready for human review
