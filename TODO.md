# TODO — AI Academic Advisor

> Working notes for future Claude Code sessions. Last updated 2026-07-05.
> Read top-to-bottom once before starting. It captures the current state, the single
> biggest architectural gap, and playbooks for the jobs that matter next.

---

## 0. TL;DR for a cold start

Three moving parts:

1. `catalog_ingestion/` — Postgres pipeline (Acalog scraper + PurdueIO importer). DB runs
   in a container (`catalog-ingestion-postgres`, host port **5433**). The undergrad catalog
   is now populated (see counts below).
2. `backend/` (FastAPI) + `clients/web/` (Next.js) — the app. The `/academic/*` endpoints
   and the RAG advisor (`/advisor/ask`) read the real DB. **But the planner and the web UI
   still run on 8 hardcoded seed courses** (`backend/app/data/courses.json`).
3. **RAG advisor** — `/advisor/ask` embeds the question, retrieves the nearest catalog
   chunks from the pgvector `academic_rules` table, and grounds the local model on them.
   The code is done; the table still needs to be populated (see §1 — the immediate task).

**Two keystones remain:** (a) **activate RAG** (§1, quick, in progress) and (b) **bridge the
planner to the real DB** (§5, the deep one). Everything else supports these.

### Current state (verify with `cd catalog_ingestion && make counts`)
| table                | rows        | notes                                                  |
|----------------------|-------------|--------------------------------------------------------|
| catalog_years        | 13          | all discovered                                         |
| courses              | 10,606      | full PurdueIO import, **2026-2027 only**               |
| subjects             | 208         |                                                        |
| **programs**         | **~931**    | undergrad 2026-2027 — main crawl done (was 1)          |
| requirement_groups   | many        | from the crawl                                         |
| requirement_options  | many        | most linked to a course                                |
| colleges             | populated?  | now written as a crawl side-effect — **verify** (§2.2) |
| **departments**      | **0**       | **no code writes this table** (§2.3)                   |
| prerequisite_rules   | ~22         | Purdue does not publish prereqs in the catalog (§2.5)  |
| **academic_rules**   | **0 / none**| RAG vectors — **not created/populated yet** (§1)       |

Grad (MS/PhD) programs are **not** ingested yet — `KNOWN_PROGRAMS_NAVOIDS` is undergrad-only.
The demo profile is "MS Computer Science", so this gap is visible (§2.1).

---

## 1. ACTIVATE THE RAG ADVISOR — immediate next step

`/advisor/ask` is fully wired (embed → pgvector search → grounded generate, with `sources`),
but it returns "no rule on file" until two things are true: the DB has the `vector` extension,
and `academic_rules` is populated. Both are pending.

**1a. Postgres must have pgvector.** `docker-compose.yml` already targets
`pgvector/pgvector:pg16` (same PG16 on-disk format as `postgres:16-alpine`, so the
`catalog-pg-data` volume is reused as-is). As of this writing the *running* container was
still the old alpine image mid-recreate — finish the swap:
```bash
cd catalog_ingestion
podman compose up -d --force-recreate postgres   # or: podman compose down && podman compose up -d postgres
make psql -c "CREATE EXTENSION IF NOT EXISTS vector;"   # sanity check; ingestion also does this
```
Confirm: `make psql -c "SELECT 1 FROM pg_available_extensions WHERE name='vector';"` → 1 row.

**1b. Pull the embedding model** (separate from the chat model; must be reachable by the guard):
```bash
ollama pull nomic-embed-text
```

**1c. Ingest the catalog into the vector store** (embeds ~10.6k courses + program requirement
blocks; slow, but idempotent — re-runs only embed new/changed chunks):
```bash
cd backend && source .venv/bin/activate
python -m app.services.rag.ingest_catalog --dry-run     # preview chunks, no model calls
python -m app.services.rag.ingest_catalog               # the real run
```
Then `curl -X POST localhost:8000/advisor/ask -H 'Content-Type: application/json' \
  -d '{"question":"I want to major in CS — what should I take first?"}'` should return an
answer plus `sources`.

**Follow-ups once it works:**
- Consider calling `store.ensure_schema()` on app startup (best-effort) so a fresh DB has the
  table without a manual step — but don't couple app boot to DB writes if the DB may be down.
- Chunking is one chunk per course + one per requirement block (`rag/ingest_catalog.py`).
  Tune `MAX_CHUNK_CHARS`, `RAG_TOP_K`, `RAG_MIN_SIMILARITY` (`.env`) once you can eyeball
  real retrieval quality. Big requirement blocks get clipped — split them if recall suffers.

---

## 2. SCRAPING PLAYBOOK — remaining catalog gaps

> Background: `catalog_ingestion/docs/ingestion_design.md` and `.../docs/schema.md`.
> Headless browser is mandatory (AWS WAF JS challenge on `content.php`/`preview_*`), so
> `FETCHER_BACKEND=playwright` for everything except year discovery. robots.txt mandates
> `Crawl-delay: 120`; a full year is ~959 programs × 120s ≈ 32h. Run backgrounded; the disk
> page cache makes re-runs cheap. The undergrad 2026-2027 crawl is **done** — below is what's left.

### 2.1 Graduate programs (MS/PhD) — not ingested
`KNOWN_PROGRAMS_NAVOIDS` in `discover/programs.py` has only the undergrad list
(`19: 25484`). To add grad programs:
```bash
# Inspect the "Graduate and Undergraduate Programs Lists" hub to find the grad list navoid:
podman compose run --rm ingestion fetch-url \
  "https://catalog.purdue.edu/content.php?catoid=19&navoid=25586" -o /tmp/hub.html
```
Then either add the grad navoid to `KNOWN_PROGRAMS_NAVOIDS`, or (better) generalize
`sync-programs` to iterate **both** lists — extend `PROGRAMS_LIST_LINK_TEXT_RE` /
`resolve_programs_navoid` to also match "Graduate Programs List". The rest of the pipeline
(`parse_program_page` → `parse_requirement_sections` → `ingest_program`) is unchanged and
idempotent. Verify: `make psql -c "SELECT degree_type, count(*) FROM programs GROUP BY 1;"`.

### 2.2 Colleges — verify + harden
"School" == "college" here (the API surfaces `colleges.name` as `schools`). Colleges are
written as a side-effect of `ingest_program` (`get_or_create_college`, only when
`parsed.college_name` is truthy). With ~931 programs ingested they should now be populated —
**verify** (`make psql -c "SELECT count(*), count(DISTINCT name) FROM colleges;"`).

`_find_college` (`parse/programs.py`) is still brittle (grabs the first "X of Y" match, which
can be a department or nav string). Harden it:
1. Prefer the breadcrumb / "Return to:" anchor over free-text regex.
2. Normalize names (trailing punctuation, "College of Engineering" variants).
3. Backfill/spot-check distinct `college_name` values against Purdue's ~13 colleges.

### 2.3 Departments — NOT IMPLEMENTED (needs new code)
`departments` has a model and `programs.department_id` is a real FK, but **nothing populates
it** (no `get_or_create_department`, no `department_name` parse). To implement:
1. **Parse**: add `department_name: str | None` to `ParsedProgram` + a `_find_department`
   helper scoped to the breadcrumb/sidebar (so it doesn't collide with the college match).
2. **Ingest**: add `get_or_create_department(session, *, catalog_year_id, college_id, name)`
   and set `program.department_id` in both insert/update branches of `ingest_program`.
3. **Expose** (optional): add `departments` to `AcademicFacetResponse` + a query in
   `fetch_academic_facets` (mirror colleges/subjects). Keep the HTTP contract additive.
4. **Re-run** `make sync-programs YEAR=2026-2027` (idempotent) to backfill.
> Store `NULL` when a page has no clean department — don't guess.

### 2.4 Multi-year courses & history
Courses exist for **2026-2027 only**. To add a year:
```bash
make load-courses YEAR=2025-2026     # PurdueIO import (fast)
make sync-programs YEAR=2025-2026    # programs+requirements (slow crawl)
```
Archived years may use different navoids — rely on `resolve_programs_navoid`, don't hardcode.

### 2.5 Prerequisites (documented gap — don't chase blindly)
Purdue's Acalog course pages contain **no** prereq text (verified on CS 25100); the ~22 rules
come from rare pages that do. Real prereqs live in Banner / myPurduePlan — a **separate
sourcing project**. The parser correctly stores `prerequisites_raw=NULL` rather than inventing
structure. If the RAG advisor or planner needs prereqs, flag it; don't fabricate.

---

## 3. LOCAL LFM2 AGENT — write & revise plans (the big AI feature)

> Goal: a local model (Liquid **LFM2** via Ollama) that (a) drafts a plan from a profile +
> requirements and (b) revises it from free-text feedback ("less theory-heavy", "6 cr/term"),
> while the **deterministic planner stays the source of truth** for legality.

### 3.1 Current LLM wiring (what exists)
- `ollama_client.py` — `generate()` (single-shot `/api/generate`, `stream=False`, strips
  `<think>` blocks) and `embed()` (for RAG). Enforces the **local-only endpoint guard**
  (localhost, LAN/private, Tailscale CGNAT, docker names). Keep the guard; no cloud fallback.
- `config.py` — chat model `lfm2.5:latest` (note: `lfm2.5:8b` does not resolve on the target
  host), embed model `nomic-embed-text`, `ollama_local_only=True`.
- `routes.py` — `POST /advisor/ask` (RAG, the main path, returns `sources`),
  `POST /advisor/explain-plan` (read-only narration of a supplied plan), `GET /health/ollama`.
  The web `AdvisorChat` calls `/advisor/ask` and renders the retrieved sources.

### 3.2 Architecture: LLM proposes, deterministic planner disposes
Do **not** let the model emit a final schedule — it will hallucinate prereqs and credit math.
```
profile + program requirements (DB)
   → generate_plan()  → baseline legal plan (planner.py)
   → LLM given (profile + baseline + requirement context + feedback) emits a JSON edit proposal
   → apply proposal → re-run generate_plan() → re-validate
        ├─ legal   → return revised plan + rationale
        └─ illegal → feed planner warnings back to the LLM, retry (loop ≤ 2-3)
```
`SemesterPlan.warnings` (missing prereqs / term-not-offered / blocked) is exactly the text to
feed back on an illegal proposal.

### 3.3 Concrete tasks
1. **Proposal schema** in `schemas.py` (Pydantic ⇒ validated for free), e.g.
   `PlanEditProposal{ rationale, reorder[], defer[], max_credits_per_semester?, avoid_tags[] }`
   — maps onto knobs the planner already understands.
2. **`generate_json()`** in `ollama_client.py` using Ollama's `format:"json"`, then
   `PlanEditProposal.model_validate`. Reuse the existing `httpx` client + guard.
3. **`advisor_agent.py`** orchestrating the loop above (cap ~2-3 iterations).
4. **`POST /advisor/revise-plan`** `{profile, current_plan, feedback}` → agent. Mirror
   error handling (`LocalModelEndpointError`→400, Ollama failure→502).
5. **Guardrails**: keep "not an official advisor / don't invent courses/prereqs" framing; the
   model only *reorders/defers* codes the deterministic layer already validated.
6. **Web**: add `revisePlan()` to `lib/api.ts`; in `AdvisorChat`, when a message is feedback
   (vs. a question), call revise-plan and re-render `SemesterCard`s + the rationale.

---

## 4. KEYSTONE — bridge the planner to the real DB

Until this lands, the `/academic/*` endpoints and the RAG advisor know the whole catalog, but
the **planner** and the web plan view still only know 8 seed courses.

1. **Planner reads the academic DB, not `courses.json`.** Add a catalog source mapping
   `courses` (+ requirement rows) into the existing `Course` schema so `generate_plan` keeps
   its signature. Keep `courses.json` as a documented offline fixture (`services/catalog.py`).
2. **Drive plans from a real `program_id`.** Add optional `program_id` to `StudentProfile`;
   derive `remaining_courses` from that program's `requirement_options` minus completed,
   instead of the client hand-listing codes.
3. **Handle "choose N from list."** DB models selective/elective groups
   (`is_selective_option`); the planner only understands a flat required list today. Required
   blocks first, then selectives.
4. **Persist profiles/plans.** Nothing is saved. Add `students`/`plans` tables (Alembic
   migration alongside `catalog_ingestion/db/migrations/versions/`) + `POST/GET /students`.

---

## 5. Web buildout (after §4 lands)
- Program picker using `/academic/facets` + `/academic/programs` (unused in the UI);
  replace the hardcoded `demoProfile` in `app/page.tsx`.
- Make `StudentProfilePanel` editable (completed courses, start term/year, credit cap) and
  feed it into `generatePlan`.
- Course search box backed by `/academic/courses/search`.
- Requirement-progress view rendering `/academic/programs/{id}` blocks/rules.
- Remove/replace the stale `clients/web/lib/mockData.ts`.
- `RAG advisor citations` are done (`AdvisorChat` renders `sources`).

---

## 6. Running the stack (commands that exist today)
```bash
cd catalog_ingestion
make up        # start postgres + backend  (API: http://localhost:8000/docs)
make psql      # SQL shell
make counts    # row counts
make backup / make restore   # dump/load catalog_db_backup.sql.gz
make help      # every target
```
Ingestion CLI runs in the `ingestion` compose service: `podman compose run --rm ingestion <cmd>`.

Backend (no containers):
```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
python -m app.services.rag.ingest_catalog   # populate the RAG store (see §1)
```
Web: `cd clients/web && npm install && npm run dev` (http://localhost:3000).

---

## 7. Hygiene / gotchas
- Backend tests cover the planner, ollama client, RAG store, and the ingestion chunk builders
  (`tests/test_ingest_catalog.py`). Add coverage for `academic_db.py` once it backs the planner (§4).
- `explainPlan()` in `clients/web/lib/api.ts` is now unused by the UI (AdvisorChat moved to
  `/advisor/ask`). Remove it, or wire explain-plan into the plan view — don't leave it dangling.
- README "Suggested Development Order" is partly stale (RAG is built) — trim it.
- Rootless podman prints a harmless "rootless netns: kill network process: permission denied"
  on teardown — the Make `stop/down/reset` targets already work around it.
- The page cache (`catalog_ingestion/.page_cache/`) makes re-scrapes cheap/offline — don't
  delete it casually; parser bugs can be fixed and data reprocessed from cached HTML.
- pgvector `academic_rules` and the relational tables share one DB but are decoupled in code
  (`rag/store.py` has its own connection) — either can be dropped without breaking the other.
