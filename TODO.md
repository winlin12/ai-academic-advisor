# TODO — AI Academic Advisor

Working notes and next steps. Last updated 2026-07-05.

## Current State

**What works:**
- ✅ Course catalog: 10,575 courses (2026-2027)
- ✅ Deterministic planner: validates prerequisites, term offerings, credit caps
- ✅ RAG advisor: answers questions about courses with sources
- ✅ LFM2 agent: proposes plan revisions, deterministic planner validates
- ✅ Database persistence: student profiles and plans stored
- ✅ Web UI: asks questions, generates plans, revises with feedback
- ✅ API: `/plan/generate`, `/advisor/ask`, `/advisor/revise-plan`

**What's missing:**
- ❌ Degree programs: 0 rows (data lost on machine move — must re-crawl)
- ❌ Program requirements: 0 rows (depends on programs)
- ❌ Graduate programs: not ingested yet
- ❌ Prerequisite rules: 0 rows (Purdue doesn't publish; need separate source)

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

## Priority 2: Web UI Program Picker

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

## Priority 3: Course Search & Editing

**Why:** Entering completed courses should be fast.

**Scope:** ~3 hours.

**Steps:**
1. Add a searchable course input to the profile editor (debounced `GET /academic/courses/search`)
2. Allow bulk import from transcript (copy-paste list of codes)
3. Display selected courses as chips; allow removal

---

## Priority 4: Requirement Progress View

**Why:** Students should see what remains to graduate, grouped by major requirements.

**Scope:** ~5 hours.

**Steps:**
1. Fetch `GET /academic/programs/{id}` (returns requirement blocks)
2. Render as collapsible sections: "Core CS" → "Data Structures" (met), "Algorithms" (blocked by prereq)
3. Cross-reference student's completed courses and plan
4. Show which courses satisfy which requirement (why this course matters)

---

## Priority 5: Graduate Programs & Other Years

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
