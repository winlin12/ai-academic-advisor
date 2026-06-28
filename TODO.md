# TODO — AI Academic Advisor

> Working notes for future Claude Code sessions. Last updated 2026-06-28.
> Read this top-to-bottom once before starting; it captures the current state, the
> single biggest architectural gap, and step-by-step playbooks for the two jobs that
> matter next: (1) **finishing the catalog scrape** (programs, requirements, colleges,
> departments) and (2) **wiring a local Ollama / LFM2 agent** that writes and revises plans.

---

## 0. TL;DR for a cold start

There are **two stacks that don't talk to each other yet**:

1. `catalog_ingestion/` — a real Postgres pipeline (Acalog scraper + PurdueIO importer).
   The DB is **running** (`catalog-ingestion-postgres`, host port **5433**) but only
   partially populated.
2. `backend/` (FastAPI) + `clients/web/` (Next.js) — the app. The `/academic/*` endpoints
   read the real DB, **but the planner and the entire web UI ignore it** and run on 8
   hardcoded seed courses (`backend/app/data/courses.json`).

**The keystone task is bridging those two worlds** (see §4). Everything else supports it.

### Current DB row counts (verify before trusting)
```bash
cd catalog_ingestion && make counts
```
| table                 | rows   | meaning                                              |
|-----------------------|--------|------------------------------------------------------|
| catalog_years         | 13     | all years discovered ✓                               |
| courses               | 10,606 | full PurdueIO import, **2026-2027 only**             |
| subjects              | 208    | ✓                                                    |
| **programs**          | **1**  | only *Agricultural Education BS* — scrape barely ran |
| requirement_groups    | 41     | from that one program                                |
| requirement_options   | 96     | 95 linked to a course                                |
| **colleges**          | **0**  | never populated (parser gap, see §2.2)               |
| **departments**       | **0**  | **no code writes this table at all** (see §2.3)      |
| prerequisite_rules    | 22     | Purdue does not publish prereqs in the catalog       |

---

## 1. Running the stack (commands that exist today)

All data/infra commands live in `catalog_ingestion/Makefile` (rootless podman):
```bash
cd catalog_ingestion
make up        # start postgres + backend  (API: http://localhost:8000/docs)
make psql      # SQL shell into the DB
make counts    # row counts for the main tables
make backup    # dump -> catalog_db_backup.sql.gz
make restore   # load  <- catalog_db_backup.sql.gz
make help      # list every target
```
The ingestion CLI (`catalog-ingest`, defined in `catalog_ingestion/src/catalog_ingestion/cli.py`)
runs inside the `ingestion` compose service, e.g. `podman compose run --rm ingestion <cmd>`.
Make targets wrap the common ones.

Backend dev loop (without containers):
```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Web dev loop:
```bash
cd clients/web && npm install && npm run dev   # http://localhost:3000
```

---

## 2. SCRAPING PLAYBOOK — programs, requirements, colleges, departments, schools

> Background reading: `catalog_ingestion/docs/ingestion_design.md` (Phase-1 live findings)
> and `catalog_ingestion/docs/schema.md` (table reference).
>
> **Why a headless browser is mandatory:** `content.php`, `preview_program.php`, and
> `preview_course_nopop.php` return an AWS WAF JS challenge (HTTP 202) to plain HTTP
> clients. Only `index.php` is CDN-served. So `FETCHER_BACKEND=playwright` is required for
> everything except catalog-year discovery. This runs the real JS challenge — it is not a
> bypass. robots.txt mandates `Crawl-delay: 120`, honored by default
> (`crawl_delay_seconds=120.0` in `catalog_ingestion/src/catalog_ingestion/config.py`).
> **Budget ~959 programs × 120s ≈ 32 hours** for a full current-year crawl. Run it
> backgrounded and resumable; the disk page cache (`.page_cache/`) means re-runs are cheap.

### 2.1 Degrees / programs + requirements (the main crawl)

This is mostly *running a command that already works*, not writing code.

```bash
cd catalog_ingestion
# Dry run first — discovers links + parses WITHOUT writing, to sanity-check counts:
podman compose run --rm ingestion sync-programs --year 2026-2027 --limit 5 --dry-run
# Then the full crawl (long; run backgrounded, e.g. nohup or a detached run):
make sync-programs YEAR=2026-2027            # == sync-programs --year 2026-2027
```

**Pipeline (already implemented — trace it if results look wrong):**
- `discover.programs.build_programs_list_url` → `/content.php?catoid=19&navoid=25484`
  (the **flat** "Undergraduate Programs List", 959 links; note 25586 is the empty hub).
- `discover.programs.discover_program_links` → unescapes HTML entities (`&amp;poid` bug),
  extracts `(url, poid)` pairs.
- `parse.programs.parse_program_page` → `ParsedProgram` (name, degree_type, program_type,
  campus, **college_name**, total_credits).
- `parse.requirements.parse_requirement_sections` → nested `RequirementGroupData` tree
  (Acalog `div.acalog-core` blocks; heading level encodes nesting).
- `ingest.programs.ingest_program` → upserts `programs` + `requirement_groups` +
  `requirement_options` (+ `program_notes`). **Idempotent**, keyed on
  `(catalog_year_id, name, degree_type)`; re-running re-parses safely.

**Graduate programs:** the current navoid (25484) is the *Undergraduate* list. To get MS/PhD
programs (needed — the demo profile is "MS Computer Science"), find the graduate programs
list navoid for catoid 19 and add it. Use the resolver that already exists:
```bash
# Fetch the programs hub page and inspect anchors, or reuse resolve_programs_navoid():
podman compose run --rm ingestion fetch-url \
  "https://catalog.purdue.edu/content.php?catoid=19&navoid=25586" -o /tmp/hub.html
```
Then extend `KNOWN_PROGRAMS_NAVOIDS` in `discover/programs.py`, or better, generalize
`sync-programs` to iterate **both** undergrad and grad list navoids
(`resolve_programs_navoid(hub_html, catoid)` already matches link text — extend its regex
`PROGRAMS_LIST_LINK_TEXT_RE` to also catch "Graduate Programs List").

**Verify after the crawl:**
```bash
make counts
make psql -c "SELECT degree_type, count(*) FROM programs GROUP BY 1 ORDER BY 2 DESC;"
podman compose run --rm ingestion validate --year 2026-2027   # parser accuracy report
```

### 2.2 Colleges (a.k.a. "schools")

**There is no separate `schools` table — in this codebase a "school" *is* a college.**
The API surfaces `colleges.name` as `schools` (see `backend/app/services/academic_db.py`
`fetch_academic_facets`, and the `school` column on program summaries). So "scrape schools"
== "populate `colleges`".

**How colleges get written:** as a side-effect of `ingest_program`. It calls
`get_or_create_college(session, catalog_year_id, name=parsed.college_name)` **only if**
`parsed.college_name` is truthy (`ingest/programs.py:66`). `college_name` comes from
`parse/programs.py:_find_college`, which regex-searches the first 3000 chars of page text
for `(College|School|Department|Division) of <Name>` (`SCHOOL_RE`).

**Why it's 0 today:** only one program is ingested and its page didn't match `SCHOOL_RE`.
Most colleges will appear automatically once §2.1 runs across all 959 programs.

**Known parser weakness to fix:** `_find_college` is brittle —
- It grabs the *first* "X of Y" match in the page, which can be a department, a generic
  catalog-nav string, or a description sentence rather than the owning college.
- Acalog program pages usually render the owning college in a breadcrumb / sidebar
  (`div.breadcrumb`, or the "Return to:" link block), which is more reliable than free text.

**TODO:** harden `_find_college`:
1. Prefer the breadcrumb / "Return to:" anchor (fetch a known program page with
   `fetch-url -o` and inspect the DOM around the college link).
2. Normalize names (strip trailing punctuation, collapse "College of Engineering" variants).
3. Backfill: after the full crawl, run a query to list distinct `college_name` values and
   spot-check against Purdue's ~13 colleges before trusting the facet list.

### 2.3 Departments — **NOT IMPLEMENTED, needs new code**

`departments` has a model (`db/models.py` `Department`) and `programs.department_id` is a
real FK, **but nothing populates it**: `ingest_program` imports `College` only and never
constructs a `Department`; `parse_program_page` never extracts a department. This is a
genuine code gap, not an unrun command.

**TODO — implement department ingestion:**
1. **Parse** the department on the program page. Acalog program pages list the owning
   department near the college (often `School of X` / `Department of Y` in the breadcrumb
   or the heading block). Add a `department_name: str | None` field to `ParsedProgram`
   (`parse/programs.py`) and a `_find_department(s)` helper analogous to `_find_college`,
   but scoped to the breadcrumb/sidebar so it doesn't collide with the college match.
2. **Ingest** it. Add `get_or_create_department(session, *, catalog_year_id, college_id,
   name)` to `ingest/programs.py` (mirror `get_or_create_college`; the `Department` model
   already has `catalog_year_id`, `college_id`, `name`, `source_page_id`). Set
   `program.department_id` in both the insert and update branches of `ingest_program`.
3. **Expose** it (optional, later). If the app needs department facets/filters, add a
   `departments` list to `AcademicFacetResponse` (`backend/app/models/schemas.py`) and a
   query in `fetch_academic_facets` (mirror the `colleges`/`subjects` blocks). Keep the
   existing HTTP contract additive so the web client doesn't break.
4. **Re-run** `make sync-programs YEAR=2026-2027` (idempotent) to backfill, then
   `make psql -c "SELECT count(*) FROM departments;"`.

> Note: Acalog does not always print a clean department on every program page. Store
> `department_name=NULL` rather than guessing — same discipline the prereq parser uses
> (`ingestion_design.md` §"Prerequisites are NOT in the Purdue catalog").

### 2.4 Multi-year courses & history

Courses currently exist for **2026-2027 only**. `courses`/`requirements` are versioned per
`catalog_year_id` on purpose ("what were the requirements when I enrolled?"). To add a year:
```bash
make load-courses YEAR=2025-2026          # PurdueIO import (fast, no scraping)
make sync-programs YEAR=2025-2026         # programs+requirements (slow crawl)
```
Caveat from `ingestion_design.md`: archived years may use different navoids; rely on
`resolve_programs_navoid` rather than hardcoding. PurdueIO is term-based (not catalog-year
versioned), so historical course *definitions* reuse current ones by code.

### 2.5 Prerequisites (documented gap — don't chase blindly)

Purdue's Acalog course pages contain **no** prerequisite text (verified on CS 25100). Only
22 prereq rules exist, from the rare pages that do. Purdue keeps prereqs in Banner /
myPurduePlan. If the planner needs real prereqs, that's a **separate sourcing project**
(scheduling system or a supplementary dataset) — flag it, don't fabricate. The parser
correctly stores `prerequisites_raw=NULL` rather than inventing structure.

---

## 3. LOCAL OLLAMA / LFM2 AGENT — write plans & suggest on feedback

> Goal: a local model (Liquid **LFM2**, served by Ollama) that (a) drafts a semester plan
> from a student profile + degree requirements and (b) revises it in response to free-text
> feedback ("make my last year less theory-heavy", "I can only do 6 credits/term"),
> while the **deterministic planner stays the source of truth** for legality
> (prereqs, term offering, credit caps).

### 3.1 Current LLM wiring (what exists)

- `backend/app/services/ollama_client.py` — `OllamaClient.generate(system_prompt,
  user_prompt)` does a single-shot `POST /api/generate` (`stream=False`). It enforces a
  **local-only endpoint guard** (`is_local_model_endpoint`): localhost, LAN/private IPs,
  Tailscale CGNAT, `host.docker.internal`, docker service names. Public hosts are rejected
  so test calls can't spend cloud API money. Keep this guard.
- `backend/app/core/config.py` — `ollama_base_url` (`http://localhost:11434`),
  `ollama_model` (`qwen2.5-coder:7b`), `ollama_local_only=True`.
- `backend/app/api/routes.py` — `POST /advisor/explain-plan` (explanation only) and
  `GET /health/ollama`. The web `AdvisorChat` component calls explain-plan.

### 3.2 Serve LFM2 in Ollama

```bash
# On the machine running Ollama (Mac / LAN box / GPU PC):
ollama pull lfm2.5:8b            # the model this project targets (~8B params)
ollama serve                     # if not already running (default :11434)
ollama list                      # confirm the tag resolved as expected
```
Point the backend at it (in `backend/.env` or root `.env`):
```env
OLLAMA_BASE_URL=http://<host-or-tailscale-ip>:11434
OLLAMA_MODEL=lfm2.5:8b
OLLAMA_LOCAL_ONLY=true
```
Verify: `curl http://localhost:8000/health/ollama` → `{"ok": true, ...}`.

> `lfm2.5:8b` is mid-sized (~8B) — capable of structured drafting but heavy enough on a
> local box that latency matters: keep prompts short and the retry loop bounded (§3.3,
> cap ~2-3 iterations). The system prompt in explain-plan already notes "keep answers
> concise" for local models — keep that. **Do not assume robust native tool-calling**; the
> reliable pattern below uses **structured-JSON prompting + server-side validation**
> (Ollama's `format: "json"`), not Ollama's tool API.

### 3.3 Architecture: LLM proposes, deterministic planner disposes

Do **not** let the model emit a final schedule directly — it will hallucinate prereqs and
credit math. Instead:

```
StudentProfile + Program requirements (from DB)
        │
        ▼
generate_plan()  ──►  baseline legal plan (deterministic, planner.py)
        │
        ▼
LLM (LFM2) given: profile + baseline plan + requirement context + user feedback
        │  emits a JSON "plan edit proposal" (move/swap/defer courses, change caps)
        ▼
apply proposal → re-run generate_plan() → re-validate (prereqs/terms/credits)
        │
        ├─ legal?  → return revised plan + LLM's natural-language rationale
        └─ illegal? → feed the planner's warnings back to the LLM, ask it to retry (loop ≤ N)
```

The deterministic planner (`backend/app/services/planner.py`) already produces
`SemesterPlan.warnings` (missing prereqs, term-not-offered, blocked courses) — that warning
text is exactly what you feed back to the model on an illegal proposal.

### 3.4 Concrete backend tasks

1. **Define the proposal schema** in `backend/app/models/schemas.py` (Pydantic, so the
   model's JSON is validated for free), e.g.:
   ```python
   class PlanEditProposal(BaseModel):
       rationale: str
       reorder: list[str] = []                  # course codes in desired priority order
       defer: list[str] = []                    # push later / drop from near-term
       max_credits_per_semester: int | None = None
       avoid_tags: list[str] = []               # e.g. ["theory-heavy","project-heavy"]
   ```
   These map onto knobs the planner already understands (`requirement_tags`,
   `max_credits_per_semester`, candidate sort order).

2. **Add a structured-generation helper** to `ollama_client.py`, e.g.
   `async def generate_json(system_prompt, user_prompt, schema_hint) -> dict`. Use Ollama's
   `format: "json"` option on `/api/generate` to force JSON, then `PlanEditProposal.model_validate`.
   Reuse the existing `httpx` client + local-only guard; do **not** add a cloud fallback.

3. **Add an agent service** `backend/app/services/advisor_agent.py` that orchestrates the
   loop in §3.3: build context → call `generate_json` → translate proposal into planner
   inputs → `generate_plan` → if warnings, append them to the prompt and retry (cap at
   ~2-3 iterations to bound local compute) → return `{plan, rationale, warnings}`.

4. **Add an endpoint** in `routes.py`, e.g. `POST /advisor/revise-plan`
   `{profile, current_plan, feedback}` → calls the agent service. Keep the existing
   `/advisor/explain-plan` for read-only Q&A. Mirror its error handling
   (`LocalModelEndpointError` → 400; Ollama failure → 502).

5. **Guardrails (carry over the existing safety stance):** the system prompt must keep
   "You are not an official academic advisor… do not invent courses/prereqs/requirements…
   explain only from the supplied structured plan… recommend verifying with an official
   advisor." The model never invents courses — it only *reorders/defers* codes that the
   deterministic layer already validated against the catalog.

### 3.5 Web tasks for the agent

- Extend `clients/web/lib/api.ts` with `revisePlan(profile, plan, feedback)`.
- Upgrade `clients/web/components/AdvisorChat.tsx`: when the user's message is feedback
  (vs. a question), call `revise-plan`, then re-render the returned `SemesterCard`s and show
  the model's `rationale`. Today it only calls `explainPlan` and shows text.

---

## 4. KEYSTONE — bridge the planner to the real DB (do this alongside §2/§3)

Until this is done, the polished `/academic/*` endpoints stay invisible to users and the
planner only knows 8 seed courses.

1. **Planner reads the academic DB, not `courses.json`.** Add a catalog source that maps
   `courses` (+ requirement/prereq rows) into the existing `Course` schema
   (`backend/app/models/schemas.py:Course`) so `generate_plan` keeps its signature. Keep
   `courses.json` as a documented test fixture / offline fallback (`services/catalog.py`).
2. **Drive plans from a real `program_id`.** Add optional `program_id` to `StudentProfile`;
   derive `remaining_courses` from that program's `requirement_options` minus
   `completed_courses`, instead of the client hand-listing codes.
3. **Handle "choose N from list".** The DB models selective/elective groups
   (`is_selective_option`); the planner currently only understands a flat required list.
   Required blocks first, then selectives.
4. **Persist profiles/plans.** Nothing is saved today. Add a `students`/`plans` table
   (Alembic migration alongside `catalog_ingestion/db/migrations/versions/`) + `POST/GET
   /students`, so a plan survives a refresh.

---

## 5. Web buildout (after §4 lands)

- Program picker panel using `/academic/facets` + `/academic/programs` (neither is used in
  the UI yet); replaces the hardcoded `demoProfile` in `clients/web/app/page.tsx`.
- Make `StudentProfilePanel` editable (completed courses, start term/year, credit cap) and
  feed it into `generatePlan`.
- Course search box backed by `/academic/courses/search`.
- Requirement-progress view rendering `/academic/programs/{id}` blocks/rules.
- Remove/replace the stale `clients/web/lib/mockData.ts`.

---

## 6. Hygiene / gotchas

- Backend tests (`backend/tests/`) cover only the planner + ollama client — add coverage
  for `academic_db.py` once it backs the planner.
- Drop the unused legacy `academic_db_path` setting in `backend/app/core/config.py` (the
  Postgres URL is the real source now).
- README "Suggested Development Order" is stale vs. what's actually built; update it.
- Rootless podman on this host prints a harmless "rootless netns: kill network process:
  permission denied" on teardown — Make `stop/down/reset` already work around it.
- The page cache (`catalog_ingestion/.page_cache/`) makes re-scrapes cheap and offline —
  don't delete it casually; parser bugs can be fixed and data reprocessed from cached HTML.
