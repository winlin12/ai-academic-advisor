# TODO — AI Academic Advisor

Working notes and next steps. Last updated 2026-07-05.

## Current State

**What works:**
- ✅ Course catalog: 10,575 courses (2026-2027)
- ✅ Deterministic planner: validates prerequisites, term offerings, credit caps
- ✅ RAG advisor: answers questions about courses with sources
- ✅ LFM2 agent: proposes plan revisions, deterministic planner validates
- ✅ Database persistence: `students`/`plans` tables + API routes exist (`POST /students`, `POST /students/{id}/plans`)
- ✅ Web UI: asks questions, generates a plan, revises it with feedback
- ✅ API: `/plan/generate`, `/advisor/ask`, `/advisor/revise-plan`

**What's missing / rough:**
- ❌ Degree programs: 0 rows (data lost on machine move — must re-crawl)
- ❌ Program requirements: 0 rows (depends on programs)
- ❌ Graduate programs: not ingested yet
- ❌ Prerequisite rules: 0 rows (Purdue doesn't publish; need separate source)
- ❌ **No direct plan editing.** You cannot move, add, or remove a single course in a specific semester. The only two ways to change a plan are full regenerate (from the hardcoded demo profile) or LLM-mediated free-text revision, which is limited to the knobs in `PlanEditProposal` (reorder / defer / avoid-tags / credit cap) — see [Priority 2](#priority-2-direct-plan-editing).
- ❌ **No database browsing/admin UI.** The only way to look inside the DB is `make psql` or `make counts` from a terminal — there's no pgAdmin/Adminer container and no admin route in the API. See [Priority 6](#priority-6-database-browsing--admin-visibility).
- ⚠️ **API surface is inconsistent.** One flat, untagged `APIRouter` mixes RPC-style routes (`/plan/generate`, `/advisor/ask`) with REST-style ones (`/academic/programs/{id}`, `/students/{id}`), with no versioning. `/catalog/courses` is a vestigial route that reads a bundled JSON fixture (`app/services/catalog.py`) instead of the real database, duplicating `/academic/courses/search`. `ExplainPlanRequest.plan` is an untyped `dict` instead of `PlanResponse`. The `students`/`plans` persistence routes are implemented but the web client never calls them. See [Priority 5](#priority-5-api-surface-cleanup).
- ⚠️ **UI is a single hardcoded demo profile**, one flat orange/stone color theme, no navigation between views, and a profile panel that only ever displays 3 read-only fields (name/degree/target graduation) with a single "Regenerate Plan" button. See [Priority 3](#priority-3-web-ui-program-picker) and [Priority 7](#priority-7-ui-visual-pass).

**Database state** (`make counts` to verify):

| Table | Rows | Status |
|-------|------|--------|
| courses | 10,575 | ✅ Full 2026-2027 import |
| subjects | 208 | ✅ Ready |
| programs | 0 | ⚠️ **Needs crawl** |
| requirement_groups | 0 | ⚠️ **Blocked on programs** |
| requirement_options | 0 | ⚠️ **Blocked on programs** |
| colleges | 0 | ⚠️ Side-effect of program crawl |
| departments | 0 | Not implemented |
| academic_rules (RAG) | 10,575 | ✅ All courses embedded |

**Impact:** Program-driven plans and major-specific RAG answers don't work yet.

---

## Priority 1: Re-crawl Degree Programs (Blocker)

**Why:** Without programs and requirements, major-specific questions fail. The data was lost when the machine changed. This unblocks many web features.

**Scope:** ~959 undergrad programs × 120s crawl delay ≈ 32 hours. **Start backgrounded; use page cache on retries.**

```bash
cd catalog_ingestion
docker compose build ingestion   # ensure Playwright image is built (~2 GB)
make sync-programs YEAR=2026-2027 CONCURRENCY=1
make backup                      # save catalog_db_backup.sql.gz when done
```

Then backfill requirement-block RAG chunks:
```bash
cd backend && source .venv/bin/activate
python -m app.services.rag.ingest_catalog  # idempotent; adds requirement chunks
```

**Verification:**
```bash
cd catalog_ingestion && make counts  # should show programs > 0
```

**Contingency:** If the old machine still has a Postgres backup or page cache, it's faster to restore or copy the cache.

---

## Priority 2: Direct Plan Editing

**Why:** Right now a generated plan can only be replaced wholesale (regenerate) or nudged indirectly through the LLM's limited proposal knobs (reorder/defer/avoid-tags/credit-cap via `/advisor/revise-plan`). There is no way to grab one course and move it to a different semester, drop a single course, or manually add one — which is the single most-requested "edit my plan" interaction.

**Scope:** ~6-8 hours.

**Steps:**
1. Add a small, deterministic edit API (e.g. `POST /plan/edit` taking `{plan, op: "move"|"add"|"remove", course_code, target_semester}`) that re-runs the planner's validation (prereqs, term offerings, credit caps) against the edited layout and returns updated warnings — no LLM involved, this is just direct manipulation.
2. In `SemesterCard`, make each `CoursePill` support drag-and-drop (or a simple "move to..." dropdown) between semesters, plus a remove (×) affordance.
3. Add an "Add course" search box per semester (reuse `GET /academic/courses/search`).
4. Wire edits through the existing (currently unused) `POST /students/{id}/plans` so an edited plan actually persists — see [Priority 5](#priority-5-api-surface-cleanup) for cleaning up that route first if needed.

**Test:** Can drag CS381 from semester 2 to semester 3, see credit totals and warnings update immediately, and reload the page without losing the edit.

---

## Priority 3: Web UI Program Picker

**Why:** The web UI hardcodes a demo profile. Users need to pick their actual major.

**Scope:** ~4 hours once programs are loaded.

**Steps:**
1. Replace hardcoded `demoProfile` in `clients/web/app/page.tsx` with a program selector
2. Call `GET /academic/facets` to list colleges/majors
3. Fetch requirements with `GET /academic/programs/{id}`
4. Build a stepper: college → major → term → completed courses
5. Feed the real profile into `generatePlan()`

**Test:** Can create a CS major profile and see CS-specific plans.

---

## Priority 4: Course Search & Editing

**Why:** Entering completed courses should be fast.

**Scope:** ~3 hours.

**Steps:**
1. Add a searchable course input to the profile editor (debounced `GET /academic/courses/search`)
2. Allow bulk import from transcript (copy-paste list of codes)
3. Display selected courses as chips; allow removal

---

## Priority 5: API Surface Cleanup

**Why:** The API "looks horrible" in its current state — one flat, untagged `APIRouter` (`backend/app/api/routes.py`) mixing RPC-style verbs (`/plan/generate`, `/advisor/ask`, `/advisor/revise-plan`) with REST resources (`/academic/programs/{id}`, `/students/{id}`), no versioning, and at least one dead/duplicate route.

**Scope:** ~4-5 hours.

**Steps:**
1. Group routes with FastAPI `tags=[...]` (or split into per-domain routers: `academic`, `planning`, `advisor`, `students`) so `/docs` reads as organized sections instead of one long list.
2. Remove or redirect `/catalog/courses` (`app/services/catalog.py`, a bundled JSON fixture) — it duplicates `/academic/courses/search` against the real DB and is confusing dead weight.
3. Type `ExplainPlanRequest.plan` as `PlanResponse` instead of a bare `dict` (schemas.py:161-163).
4. Decide whether `/students` + `/students/{id}/plans` are worth wiring into the UI (see [Priority 2](#priority-2-direct-plan-editing)) or removing if the direction changes — right now they're implemented but orphaned.
5. Consider a consistent `/v1` prefix before more routes accumulate.

---

## Priority 6: Database Browsing & Admin Visibility

**Why:** There's currently no way to see or navigate the database outside a terminal — no pgAdmin/Adminer, no admin page, no table browser. `make psql` / `make counts` work but require dropping to a shell.

**Scope:** ~1-2 hours for a browser-based DB client; more if building custom admin views.

**Steps (pick one):**
1. **Fastest:** add an [Adminer](https://www.adminer.org/) (or pgAdmin) service to `catalog_ingestion/docker-compose.yml` pointed at the `postgres` service, exposed on its own port — gives full table browsing/editing for free.
2. **More integrated:** build a minimal `/admin` page in `clients/web` backed by a few read-only API routes (list students, list saved plans, program/requirement counts) — less powerful than Adminer but stays inside the app's own auth boundary if one is ever added.

**Test:** Can view `students`, `plans`, `programs`, and `courses` row contents without opening a terminal.

---

## Priority 7: UI Visual Pass

**Why:** The UI currently reads as messy/bland — a single flat orange/stone gradient theme applied uniformly everywhere, no visual hierarchy beyond a 3-column grid, no navigation between views, and `StudentProfilePanel` only ever renders 3 static read-only fields with one "Regenerate Plan" button.

**Scope:** ~4-6 hours; depends on how far the redesign goes.

**Steps:**
1. Establish a real design system (spacing/type scale, a second accent color for distinguishing warnings/success/info beyond amber-vs-emerald pills) rather than one repeated gradient class.
2. Add actual navigation/structure once there's more than one view (profile creation, plan, admin — Priorities 2, 3, 6) instead of everything crammed into a single page's 3-column grid.
3. Make `StudentProfilePanel` reflect real, editable profile data once Priority 3 lands, instead of 3 hardcoded display fields.

---

## Priority 8: Requirement Progress View

**Why:** Students should see what remains to graduate, grouped by major requirements.

**Scope:** ~5 hours.

**Steps:**
1. Fetch `GET /academic/programs/{id}` (returns requirement blocks)
2. Render as collapsible sections: "Core CS" → "Data Structures" (met), "Algorithms" (blocked by prereq)
3. Cross-reference student's completed courses and plan
4. Show which courses satisfy which requirement (why this course matters)

---

## Priority 9: Graduate Programs & Other Years

**In Progress:**
- Undergrad programs: blocked on Priority 1
- Grad programs (MS/PhD): requires adding navoid to `catalog_ingestion/discover/programs.py`
- Past years (2024-2025, etc.): requires `make load-courses YEAR=2025-2026` + crawl

---

## Technical Reference: Architecture & Design

### Planner Architecture: AI proposes, planner validates

**Never let the model emit a final schedule** — it will hallucinate prereqs and credits.

```
Student Profile + Requirements (from DB)
         ↓
  Deterministic Planner
         ↓
     Legal Plan (prereqs checked, credits capped, terms offered)
         ↓
  LLM proposes edits (reorder, defer, cap credits)
         ↓
  Planner re-validates edit
  ├─ legal   → return plan + rationale
  └─ illegal → loop (retry up to 3×, feed back warnings)
```

The planner is the source of truth. No hallucinated courses, no broken prerequisites.

### RAG Advisor: Semantic Search + Grounding

- `POST /advisor/ask` embeds the question, retrieves top-K similar course/requirement chunks from pgvector
- Returns cited sources so answers are verifiable
- Chunking strategy: one chunk per course + one per requirement block
- Tuning knobs: `MAX_CHUNK_CHARS`, `RAG_TOP_K`, `RAG_MIN_SIMILARITY` in `.env`

### Database Schema

**Core tables:**
- `courses` (10,575) — course catalog with credit hours, descriptions
- `programs` (0 — needs crawl) — degree programs
- `requirement_groups` (0) — major-level requirement blocks (core, electives, etc.)
- `requirement_options` (0) — courses that satisfy a requirement group
- `academic_rules` (10,575 RAG vectors) — pgvector embeddings for semantic search
- `students`, `plans` — app-side persistence (JSONB)

**Key constraints:**
- Purdue publishes **no prerequisites** in Acalog (see Priority 1 notes)
- Courses have no term offerings per se (treated as always offered)
- Some crawl data is cached in `.page_cache/` — reuse it for parser fixes

### Scraping & Ingestion

**Makefile targets** (in `catalog_ingestion/`):

```bash
make db-init                        # Create schema, indexes
make load-courses YEAR=2026-2027    # PurdueIO import (fast)
make sync-programs YEAR=2026-2027   # Crawl programs (slow, headless browser required)
make backup / make restore          # Dump/load full DB with RAG vectors
make counts                         # Check row counts per table
make psql                           # SQL prompt
```

**robots.txt:** Crawl-delay 120s; full year ≈ 32 hours. Use page cache for retries.

**Headless browser:** Required for AWS WAF JS challenges on content.php. Uses Playwright.

---

## Completed Work

### 1. RAG Advisor (✅ Done)
- All 10,575 courses embedded in pgvector
- Similarity-ranked retrieval working
- Answers cited with sources
- Pending: requirement-block chunks (blocked by program crawl)

### 2. LFM2 Revise-Plan Agent (✅ Done)
- Proposes edits (reorder, defer, credit cap)
- Deterministic planner validates
- Retry loop for invalid proposals
- Web UI integration via ask/revise toggle

### 3. Planner–DB Bridge (✅ Done)
- Planner reads real catalog courses
- Program-driven plan selection (when programs exist)
- Choose-N selective logic
- Student profile/plan persistence

### 4. Course Catalog Import (✅ Done)
- 10,575 courses for 2026-2027
- PurdueIO importer
- Can import additional years (fast)

---

## Development Commands

**Backend tests:**
```bash
cd backend && pytest                      # run all tests
pytest -xvs tests/test_planner.py        # specific test file
```

**Database:**
```bash
cd catalog_ingestion
make psql                                 # SQL prompt
make counts                               # row counts by table
```

**Debugging:**
- API docs at http://localhost:8000/docs
- Ollama health check: `curl http://localhost:8000/health/ollama`
- Catalog DB: `postgres://catalog:catalog@localhost:5433/catalog_ingestion`

---

## Known Gotchas

- **Calendar year transitions:** `planner.next_term` correctly crosses year boundaries (fall 2026 → spring 2027)
- **Page cache:** `catalog_ingestion/.page_cache/` speeds up retries — don't delete casually
- **RAG + planner:** Decoupled in code (`rag/store.py` has own connection) — either can be reset independently
- **Offline fallback:** `backend/app/data/courses.json` used when DB is down or unknown codes
- **Prerequisite gap:** Purdue doesn't publish prerequisites in Acalog; real prereqs in Banner/myPurduePlan (separate sourcing needed)
- **Fixed (2026-07-05):** `planner.next_term` now correctly crosses year boundaries (fall 2026 → spring 2027)
- **Removed:** `clients/web/lib/mockData.ts` and unused `explainPlan()` client code
