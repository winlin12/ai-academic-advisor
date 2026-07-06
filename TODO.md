# TODO — BoilerAdvisor

Working notes and next steps. Last updated 2026-07-06 (renamed from "AI Academic Advisor" to **BoilerAdvisor** the same day).

## Current State

**What works:**
- ✅ Course catalog: 10,575 courses (2026-2027)
- ✅ Deterministic planner: validates prerequisites, term offerings, credit caps
- ✅ RAG advisor: answers questions about courses with sources
- ✅ LFM2 agent: proposes plan revisions, deterministic planner validates
- ✅ Web UI: onboarding profile form (no hardcoded demo profile), plan generation, direct editing (move/remove per course, add via debounced catalog search), advisor ask/revise chat
- ✅ **Plan persistence, wired end-to-end**: onboarding calls `POST /v1/students`; every accepted edit/revision/regeneration autosaves via `POST /v1/students/{id}/plans`; a reload restores the newest saved plan (student id in `localStorage`). Save-status chip surfaces Saved/Saving/failed/DB-down.
- ✅ **Database browsing**: read-only `/admin` page (table counts + paged rows via `GET /v1/admin/*`, whitelisted tables only) and an Adminer container (`make adminer` → http://localhost:8081) for full editing.
- ✅ **Purdue black & gold UI** ("BoilerAdvisor"): design tokens in `globals.css`, top navigation (Planner / Database), onboarding hero, dark theme throughout. Root cause of the old "unstyled" look: Tailwind was never compiling — `tailwind.config.ts` + `postcss.config.mjs` didn't exist; they do now.
- ✅ API: versioned `/v1` prefix with tagged domain routers (`academic`, `planning`, `advisor`, `students`, `admin`); `/v1/plan/generate`, `/v1/plan/edit`, `/v1/advisor/ask`, `/v1/advisor/revise-plan`
- ✅ Direct plan editing: `POST /v1/plan/edit` (`services/plan_editor.py`) — deterministic move/add/remove with placement-preserving re-validation, no LLM

**What's missing / rough:**
- ⏳ Degree programs: **crawl in progress** (2026-07-06; ~32h run). Counts below are a snapshot mid-crawl.
- ❌ Graduate programs: not ingested yet
- ❌ Prerequisite rules: 0 rows (Purdue doesn't publish; need separate source)
- ⚠️ **No program/major picker in the UI** — degree program is free text until the crawl fills `programs`; then build the picker (Priority 3).
- ⚠️ **"Edit profile" creates a new student row** — there's no `PUT /v1/students/{id}`, so editing re-creates the student and re-points `localStorage`. Old rows accumulate harmlessly; add an update route to fix properly.
- ⚠️ Saved-plan history is write-only from the UI — every save appends a `plans` row, but the UI only ever loads the newest one. No history browser/restore yet (rows are visible in `/admin`).

**Database state** (`make counts` to verify — snapshot taken mid-crawl 2026-07-06):

| Table | Rows | Status |
|-------|------|--------|
| courses | 10,575 | ✅ Full 2026-2027 import |
| subjects | 208 | ✅ Ready |
| programs | 10+ | ⏳ **Crawl running** (~959 total expected) |
| requirement_groups | 316+ | ⏳ Filling as programs land |
| requirement_options | 764+ | ⏳ Filling as programs land |
| colleges | 0 | ⚠️ Side-effect of program crawl |
| departments | 0 | Not implemented |
| academic_rules (RAG) | 10,575 | ✅ All courses embedded (requirement chunks pending crawl) |

**Impact:** Program-driven plans and major-specific RAG answers unlock as the crawl completes (then re-run the RAG ingest for requirement chunks).

---

## Priority 1: Re-crawl Degree Programs (⏳ in progress 2026-07-06)

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

## Priority 2: Direct Plan Editing (✅ done 2026-07-06, including persistence)

**Why:** A generated plan could only be replaced wholesale (regenerate) or nudged indirectly through the LLM's limited proposal knobs. Now a student can grab one course and move/add/remove it directly — and it sticks.

**Done:**
1. ✅ `POST /v1/plan/edit` (`backend/app/services/plan_editor.py` + `api/routers/planning.py`) takes `{plan, operation: "move"|"add"|"remove", course_code, target_semester, profile?}`, applies the edit, and re-validates the layout placement-preservingly (prereqs via `planner.prereqs_satisfied`, term offerings, credit caps) — no LLM involved. Tests in `backend/tests/test_plan_editor.py`.
2. ✅ Each `CoursePill` has a "Move to..." semester dropdown and a remove (×) affordance; `SemesterCard` shows all warnings (not just the first).
3. ✅ Per-semester "Add course" debounced search box (`clients/web/components/AddCourseSearch.tsx`, reuses `GET /v1/academic/courses/search`).
4. ✅ Edits persist: the web client creates the student at onboarding (`POST /v1/students`, id in `localStorage`) and autosaves every accepted edit/revision/regeneration through `POST /v1/students/{id}/plans` (`persistPlan` in `clients/web/app/page.tsx`); a reload fetches `GET /v1/students/{id}` and restores `plans[0]`. Failures degrade to a visible status chip, never a blocked edit.

**Test:** Move CS536 from semester 1 to semester 2, see credit totals and warnings update immediately; reload the page and the moved layout comes back from the DB. Verified end-to-end 2026-07-06.

---

## Priority 3: Web UI Program Picker (profile form ✅ done; picker blocked on Priority 1)

**Why:** ~~The web UI hardcodes a demo profile.~~ Done 2026-07-06: the hardcoded `demoProfile` is gone — first visit shows a profile setup form (`clients/web/components/ProfileSetup.tsx`: name, degree text, start term/year, credit cap, completed/remaining courses via catalog-search chips). What's still missing is picking a real *program* so requirements drive the plan.

**Scope:** ~3 hours once programs are loaded.

**Steps:**
1. In `ProfileSetup`, replace the free-text degree field with a program selector
2. Call `GET /academic/facets` to list colleges/majors
3. Fetch requirements with `GET /academic/programs/{id}`
4. Set `program_id` on the profile so the backend derives `remaining_courses` from real requirement rows (the plumbing already exists in `planner_catalog.py`)

**Test:** Can create a CS major profile and see CS-specific plans.

---

## Priority 4: Course Search & Editing (steps 1+3 ✅ done 2026-07-06)

**Why:** Entering completed courses should be fast.

**Done:**
1. ✅ Searchable course input in the profile form (`clients/web/components/CourseChipInput.tsx`, debounced `GET /academic/courses/search`; Enter adds the raw typed code so it works offline too)
3. ✅ Selected courses render as removable chips

**Remaining:**
2. Bulk import from transcript (copy-paste list of codes)

---

## Priority 5: API Surface Cleanup (✅ done 2026-07-06)

**Done:**
1. ✅ Split the flat router into per-domain tagged routers under `backend/app/api/routers/` (`system`, `academic`, `planning`, `advisor`, `students`), aggregated in `api/routes.py`; `/docs` now reads as organized sections.
2. ✅ Removed `/catalog/courses` — the bundled fixture (`app/services/catalog.py`) survives only as the planner's documented offline fallback, no longer an API route.
3. ✅ `ExplainPlanRequest.plan` is typed as `PlanResponse` (and the response as `ExplainPlanResponse`); course search returns a typed `AcademicCourseSearchResponse`.
4. ✅ Kept `/students` + `/students/{id}/plans`: they're the persistence target for Priority 2 step 4 — and as of 2026-07-06 the web client actually uses them (onboarding + plan autosave).
5. ✅ All product routes live under `/v1`; health probes stay unversioned at the root for infra checks.

---

## Priority 6: Database Browsing & Admin Visibility (✅ done 2026-07-06 — both options)

**Done:**
1. ✅ **Adminer** service in `catalog_ingestion/docker-compose.yml` (`make adminer` → http://localhost:8081; no `depends_on`, so starting it never touches a running postgres/crawl). Full table browsing/editing.
2. ✅ **`/admin` page** in `clients/web` backed by read-only API routes: `GET /v1/admin/tables` (row counts) and `GET /v1/admin/tables/{table}` (paged rows). Strict table whitelist in `backend/app/services/admin_db.py` (`academic_rules` drops its pgvector column), rows serialized via `to_jsonb`, no write routes at all. Tests in `backend/tests/test_admin_db.py`.

**Test (passes):** Can view `students`, `plans`, `programs`, and `courses` row contents without opening a terminal.

---

## Priority 7: UI Visual Pass (✅ done 2026-07-06 — Purdue black & gold)

**Root cause found:** the old UI looked "unstyled defaults" because **Tailwind was never compiling** — the app had no `tailwind.config.ts`/`postcss.config.mjs`, so every utility class was dead text and pages rendered as browser-default HTML. Both files exist now; if the UI ever regresses to plain HTML, check them first.

**Done:**
1. ✅ Design system in `globals.css`: Purdue palette (Boilermaker Gold `#CFB991`, Field `#DDB945`, Dust `#EBD99F`, Aged `#8E6F3E` on warm black), Barlow Condensed display type + Space Grotesk body, and reusable primitives (`.card`, `.card-accent`, `.kicker`, `.btn-gold`, `.btn-ghost`, `.field`, `.boiler-stripe`). Gold is reserved for identity/action; amber = warnings, emerald = success.
2. ✅ Navigation: sticky black top bar (`components/NavBar.tsx`) with Planner/Database links and the gold "boiler stripe" rule; footer disclaimer. Views: onboarding hero, planner (3-column), `/admin`.
3. ✅ `StudentProfilePanel` shows the real profile (name/degree/start/credit cap/completed count/target graduation) with Regenerate / Edit profile / Start over actions.

**Possible future polish:** plan-history browser, drag-and-drop course moves, mobile layout audit, dark/light toggle.

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

### 5. BoilerAdvisor Rebrand, Persistence Wiring, Admin UI, Purdue Theme (✅ Done 2026-07-06)
- Renamed app to **BoilerAdvisor** (FastAPI title, web title/nav, docs)
- Web client wired to `students`/`plans` persistence: onboarding creates the student, edits autosave, reloads restore the newest plan
- Hardcoded demo profile removed — onboarding profile form with catalog-search course chips
- Read-only `/admin` DB browser (+ `GET /v1/admin/*` routes) and Adminer container (`make adminer`)
- Purdue black & gold redesign; fixed Tailwind never compiling (missing configs)

---

## Development Commands

**Backend tests:**
```bash
cd backend && pytest                      # run all tests
pytest -xvs tests/test_planner.py        # specific test file
```

**Frontend:**
```bash
cd clients/web
npm run dev                               # http://localhost:3000
npm run build                             # production build + typecheck
```

**Database:**
```bash
cd catalog_ingestion
make psql                                 # SQL prompt
make counts                               # row counts by table
make adminer                              # Adminer at http://localhost:8081
```
Read-only browsing also at http://localhost:3000/admin.

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
- **Tailwind requires its config files:** `clients/web/tailwind.config.ts` + `postcss.config.mjs` are what make Tailwind compile at all. The app shipped for weeks without them and every utility class silently rendered as unstyled HTML. Config changes need a dev-server restart.
- **CSS `@import` must be first:** the Google Fonts import in `globals.css` has to precede the `@tailwind` directives, or browsers drop it after Tailwind expands (fonts silently fall back).
- **Laptop sleep pauses the crawl:** Docker Desktop's VM suspends with the host, so the `sync-programs` container stops fetching while the Mac sleeps and resumes on wake (tenacity retries + the page cache absorb the interrupted request). For the full ~32 h run, prefer an always-on host — the Makefile auto-detects podman vs. Docker, so the stack runs unchanged on a Linux server (`make backup` → copy → `make restore`, then restart `make sync-programs`, ideally copying the page-cache volume too).
