---
type: vision
status: decomposed
decomposed_at: "2026-05-18T08:16:04.018Z"
---

# Vision: Americanize the Finike Orange Price Predictor Dashboard

## Recommendation (read first)
**Path: ONE vision → DECOMPOSE into 3 functional plan stubs.**

This seed contains three independently-shippable workstreams that share one north star
(make the dashboard legible and credible to an American Quantic grader in under 2 minutes
of use) but touch different files, different layers of the stack, and carry different
risk profiles. Decomposing keeps each plan small enough to land inside a single capstone
sprint while letting them ship in any order.

Recommended decomposition (the Vision Decomposer should produce stubs for these):

1. **plan: intro-landing-page** — Build a Finike + project introduction page that is the
   first thing a visitor sees when the deployed dashboard URL opens. Uses the existing
   `finikeliman.jpeg` and `portakal.jpeg` at project root. No data dependencies.
2. **plan: ui-string-americanization** — Replace Turkish-by-default UI labels, tab names,
   chart titles, legends, axis labels, and tooltip text (Hal, Narenciye, Hal Fiyatları,
   Tarım Bakanlığı, etc.) with American-English equivalents. Display-layer only;
   underlying column names and API payloads stay untouched.
3. **plan: news-events-en-translation** — Render the Market & Policy / Price + Policy
   Events list, event types, and event descriptions in American English. Translation
   happens at display time (preferred: extend the existing DeepSeek pipeline to emit an
   English field alongside the Turkish source) without losing source-language attribution
   or the original headline link.

A fourth workstream — a **persistent terminology glossary** (tooltip + an "About / Terms"
expander in the sidebar or footer) surfacing one-line definitions of Hal, Narenciye,
TCMB, İBB, Antalya/İstanbul Hal, Hal commission — is small enough that it can be folded
into plan (2) rather than getting its own stub. The decomposer should decide.

---

## Status
- Created: 2026-05-18
- Last Updated: 2026-05-18
- Progress: 5/5 phases complete
- Status: ready

## Phase 1: Problem Discovery
### Problem Statement
✓ The deployed Streamlit dashboard (`dashboard.py` on Render) opens directly into
operational charts with zero introductory context, and significant portions of its UI —
tab/section labels ("Hal Fiyatları", "Narenciye"), chart legends, and the entire
Market & Policy / Price + Policy Events news feed produced by the DeepSeek LLM news
layer — remain in Turkish. An American capstone grader landing on the live URL cannot
tell within the first minute (a) where Finike is, (b) what orange-orchard wholesale
prices have to do with a software engineering capstone, or (c) what the policy-event
ticker is saying about the market. This blocks the grader from evaluating the *software
engineering* of the project because they're stuck decoding the *domain*.

### Target User
✓ **Primary (graded audience):** American Quantic School of Business and Technology
capstone graders evaluating the MSc Software Engineering submission. Native English
speakers, US-based, no assumed familiarity with Turkish agriculture, Turkish wholesale
markets, or the Finike region.

✓ **Secondary (domain audience):** Future English-speaking users of the deployed
dashboard — international citrus traders, exporters, agri-analysts, and any
non-Turkish-speaking farmer or stakeholder who clicks the Render URL. The Turkish-native
domain audience (Finike orange farmers, İstanbul/Antalya Hal traders) is already served
by the existing terminology and is **not** the focus of this vision — their experience
must not regress.

### Problem Severity
✓ **High for grading:** A grader who cannot navigate the dashboard in the first
two minutes is likely to mark down on usability, documentation, and presentation
criteria even if the underlying ML and engineering are sound. This is a capstone
submission gating risk, not a nice-to-have polish item.

## Phase 2: Value Proposition
### Success Criteria
✓ A grader (or any English-speaking first-time visitor) who opens the deployed Render
URL cold can, within **two minutes and without leaving the dashboard**, answer all of:

1. *Where is Finike and what is grown there?* — Answered by an intro page that loads
   first, shows the harbor (`finikeliman.jpeg`) and orchard (`portakal.jpeg`) imagery,
   and explains in two short paragraphs that Finike (Antalya province, Turkey) is a
   coastal town and Turkey's signature region for Washington-navel-style oranges.
2. *What does this dashboard predict and for whom?* — Answered on the same intro page
   in plain American English: "Forecasts wholesale orange prices 7–90 days ahead for
   farmers, traders, exporters, and analysts."
3. *What does every label on every chart mean?* — Every Turkish word a non-Turkish
   speaker encounters in the live UI has either (a) been replaced with an
   American-English label, or (b) is followed by a one-line English gloss available
   on hover or in a visible "Terms" section.
4. *What is the news/policy ticker saying?* — Every event row in Market & Policy /
   Price + Policy Events is readable in American English (event type, description,
   direction, magnitude). Original Turkish source attribution and source-link
   remain visible for verifiability.

### Impact Scale
✓ Affects 100% of first-time English-speaking visitors to the deployed dashboard,
including the entire grading panel for the capstone submission. Estimated impact
window: from merge of these plans through the remainder of the capstone evaluation
period and indefinitely thereafter for any English-language traffic.

## Phase 3: Scope Definition
### Minimum Viable Scope
✓ Three concrete deliverables that together satisfy the success criteria:

**Workstream A — Intro / landing page**
- A new first-loaded view (sidebar route or top-of-page hero) that displays before any
  charts render.
- Uses the existing `finikeliman.jpeg` and `portakal.jpeg` from the project root.
- Contains: a 2-paragraph American-English explainer of Finike + the Finike orange,
  a 1-paragraph explainer of what the dashboard does and who it serves, a clear
  "Continue to dashboard" affordance, and an attribution line crediting Su Sarlar /
  Quantic MSc capstone.

**Workstream B — UI string Americanization (display layer)**
- Audit and translate every user-facing Turkish string in `dashboard.py` and any
  Streamlit components it renders: tab names, section headers, metric labels, chart
  titles, axis titles, legend entries, tooltip text, button labels, sidebar copy.
- Known targets (non-exhaustive): "Hal" → "Wholesale Market (Hal)" on first mention
  then "Wholesale Market" thereafter; "Narenciye" → "Citrus"; "Hal Fiyatları" →
  "Wholesale Prices"; "Antalya/İstanbul Hal" → "Antalya/İstanbul Wholesale Market".
- May include a small glossary expander or hover-tooltip mechanism so domain users
  who *want* the Turkish term still see it parenthetically.

**Workstream C — News & policy events in American English**
- Every row that surfaces in the Market & Policy / Price + Policy Events list
  (`event_type`, `description`, `impact_direction`, `impact_magnitude`, and any
  category badges) must render in American English.
- Preferred implementation: extend the existing DeepSeek `deepseek-chat` extraction
  prompt to additionally produce `event_type_en` and `description_en` fields and
  persist them alongside the Turkish originals in the events store; the UI reads the
  `_en` fields. This preserves the source-language audit trail.
- Fallback (if extending the pipeline is out of sprint budget for one plan): a
  display-time translation map for the closed vocabulary of `event_type` values
  (e.g., FROST_WARNING, EXPORT_RESTRICTION, FX_SHOCK) plus a one-time backfill of the
  English description column.

### Explicit Exclusions
✓ The following are **out of scope** for this vision and must not creep into any of
the decomposed plans:

- **No full i18n / gettext framework.** No `.po` files, no language-selector toggle,
  no runtime locale switching. American English becomes the default and only UI
  language; Turkish source values remain only as audit/attribution.
- **No re-translation of upstream API payloads.** The İBB Hal API, TCMB API, and
  Open-Meteo responses keep their original field names in raw storage. Translation
  is a display-layer concern only.
- **No changes to the prediction model.** XGBoost/LightGBM/quantile/ensemble pipelines,
  feature engineering, and training schedules are untouched. This is a presentation
  layer vision, not a modeling vision.
- **No changes to column names in `data/processed/`.** Existing snake_case Turkish-rooted
  feature names (where they exist) stay; display labels are mapped at render time.
- **No re-architecting of the DeepSeek pipeline.** The English fields can be added as
  additive output keys; the existing extraction flow is not replaced.
- **No Turkish-speaker regression.** The Turkish source description, original headline,
  and source URL for every news event remain visible (e.g., in a "show original" toggle
  or a secondary line) so the Turkish-native domain audience is not harmed.
- **No new data sources** introduced to support these changes.

### Dependencies
✓ - The two JPEGs `finikeliman.jpeg` and `portakal.jpeg` must remain at project root
(confirmed present on 2026-05-18) and be accessible from `dashboard.py` (Streamlit
serves local files via `st.image`).
- For Workstream C (preferred path): the DeepSeek API key already wired into the news
  pipeline; no new credentials.
- Render redeploy on push (already configured via `render.yaml`).

## Phase 4: Risk Assessment
### Failure Modes
✓ - **Translation drift / inaccuracy in news events.** DeepSeek-generated English
descriptions could subtly distort meaning of Turkish policy announcements. Mitigation:
preserve Turkish original alongside English, label as machine-translated, prompt the
LLM to be conservative (literal over paraphrase).
- **Intro page becomes a wall of text.** A grader who has to read 500 words before
  seeing a chart is worse off than one who sees Turkish labels. Mitigation: cap
  intro at ~150 words across two paragraphs plus the two photos.
- **Partial translation looks worse than no translation.** If half the labels are
  English and half are Turkish, the dashboard looks unfinished. Mitigation: the
  Workstream B plan must include a completeness audit (grep + manual walkthrough)
  before merge.
- **Domain users (Turkish farmers/traders) lose terminology they rely on.** Mitigation:
  the glossary surface keeps the Turkish term parenthetically on first mention of each
  domain concept.

### Unknowns
✓ - Exact count of distinct Turkish strings currently in `dashboard.py` (audit is
part of Workstream B plan).
- Whether the DeepSeek prompt extension for `*_en` fields will fit in existing token
  budgets without splitting the call.

### Assumptions
✓ - American English (not British English) is the target dialect; date formats in
new copy use US conventions (May 18, 2026) where presentational.
- Quantic graders access the dashboard at the deployed Render URL, not by running
  locally — so all changes must survive a clean Render redeploy.
- The user (Su Sarlar) is fluent in both Turkish and English and will review every
  translation in the PR; no external translator is needed.

## Phase 5: Summary

## Vision: Americanize the Finike Orange Price Predictor Dashboard

**In one sentence:** American capstone graders and English-speaking domain visitors can
understand the Finike Orange Price Predictor end-to-end within two minutes of opening
the deployed Render URL, by way of a Finike introduction landing page plus a
display-layer Americanization of every Turkish UI string and every DeepSeek-generated
news event.

**The problem:** The live dashboard opens straight into operational charts with no
context, and core UI labels (Hal, Narenciye, Hal Fiyatları) and the entire Market &
Policy news feed surface in Turkish — blocking American Quantic graders from evaluating
the software-engineering merits of the capstone in the first two minutes of use.

**For whom:** Primary — American Quantic MSc capstone graders, US-based, no Turkish.
Secondary — English-speaking citrus traders, exporters, and agri-analysts visiting the
deployed Render URL. Turkish-native domain users must not regress.

**Success looks like:** A cold English-speaking visitor can, within two minutes and
without leaving the dashboard, answer: where is Finike and what is grown there, what
does this dashboard predict and for whom, what every chart label means, and what the
news/policy ticker is saying. All four questions answered from the running dashboard,
not from external README files.

**What we are building:** Three small, independent workstreams sharing one north star —
(1) an intro/landing page with `finikeliman.jpeg` and `portakal.jpeg` plus ~150 words
of American-English context; (2) a display-layer translation pass over every Turkish
UI string in `dashboard.py` (Hal, Narenciye, Hal Fiyatları, etc.) with a parenthetical
glossary mechanism for the original terms; (3) American-English rendering of the
Market & Policy / Price + Policy Events feed, preferably by extending the DeepSeek
`deepseek-chat` extraction to emit `event_type_en` and `description_en` fields while
keeping Turkish originals as audit attribution.

**What we are NOT building:**
- A full i18n framework, locale toggle, or `.po` files
- Re-translation of upstream API payloads or renames of `data/processed/` columns
- Any change to the prediction models, feature pipeline, or training schedule
- A replacement for the DeepSeek news pipeline (additive fields only)
- Any change that removes Turkish source attribution from the news feed

**Key risk:** Machine-translated English news descriptions subtly misrepresent the
meaning of Turkish policy announcements and lead a grader (or a domain user) to a
factually wrong inference about the market. Mitigated by always rendering the Turkish
original alongside, labeling the field as machine-translated, and biasing the DeepSeek
prompt toward literal translation over paraphrase.

**RICE Score:**
- Reach: H — 100% of English-speaking dashboard visitors including the entire grading
  panel.
- Impact: H — Directly affects capstone evaluation outcome on usability, documentation,
  and presentation criteria.
- Confidence: H — Scope is bounded, dependencies are trivial (two local JPEGs + an
  existing LLM call), no model or data-pipeline risk.
- Effort: M — Three small plans, each ~1 sprint-day of work, modest test surface.

**Stakeholders:** Quantic capstone grading panel (primary). The project owner Su Sarlar
acts as the Turkish-English reviewer for every translation. No external stakeholders.

**Assumptions to test:** (a) DeepSeek can emit reliable American-English `_en` fields
within current token budgets; (b) a 150-word intro plus two photos is sufficient
context for a cold grader without becoming a wall of text.

## Discussion History

### 2026-05-18
Q: Initial seed from project owner
A: Verbatim seed captured at top of vision creation. User-supplied context covered all
four required dimensions (problem, audience, scope, success criteria implicit) plus
the two image asset names, so the Vision Advisor proceeded straight to summary with
zero follow-up questions per the hard constraint.
