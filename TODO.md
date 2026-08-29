# TODO — BoilerAdvisor

Working notes and next steps. Last updated 2026-07-21 (moved the advisor's local model from
Ollama to llama.cpp — see "llama.cpp Pivot" under Completed Work. The app had already moved
back from the Anthropic API to a local Ollama model before this; the "Anthropic API Pivot"
entry below is kept as history but is no longer the active backend).

**🎯 Next goal:** the degree-program crawl is done and the data is sitting in Postgres, but the web app doesn't expose it yet. The concrete next step is wiring the program catalog into the UI so a student can search for a degree and pull up its requirements — see **Priority 3** below. That's the next thing to hand to an agent to implement.

## Current State

**What works:**
- ✅ Course catalog: 10,607 courses (2026-2027)
- ✅ **Degree programs: full crawl complete** — 1,165 programs (2026-2027), 23,213 requirement groups, 40,758 requirement options. See Priority 1.
- ✅ Deterministic planner: validates prerequisites, term offerings, credit caps
- ✅ RAG advisor: answers questions about courses with sources; requirement-block chunks are now embedded too (see Database state). Retrieval is two-tier — exact course-code match (SQL) plus semantic search (in-process fastembed + pgvector), merged under a hard context token budget.
- ✅ Local llama.cpp agent (Qwen2.5-Coder-7B-Instruct-Q4_K_M): proposes plan revisions as a schema-enforced structured output, deterministic planner validates
- ✅ Web UI: onboarding profile form (no hardcoded demo profile), plan generation, direct editing (move/remove per course, add via debounced catalog search), advisor ask/revise chat
- ✅ **Plan persistence, wired end-to-end**: onboarding calls `POST /v1/students`; every accepted edit/revision/regeneration autosaves via `POST /v1/students/{id}/plans`; a reload restores the newest saved plan (student id in `localStorage`). Save-status chip surfaces Saved/Saving/failed/DB-down.
- ✅ **Database browsing**: read-only `/admin` page (table counts + paged rows via `GET /v1/admin/*`, whitelisted tables only) and an Adminer container (`make adminer` → http://localhost:8081) for full editing.
- ✅ **Purdue black & gold UI** ("BoilerAdvisor"): design tokens in `globals.css`, top navigation (Planner / Database), onboarding hero, dark theme throughout. Root cause of the old "unstyled" look: Tailwind was never compiling — `tailwind.config.ts` + `postcss.config.mjs` didn't exist; they do now. UI pass is done and looks substantially better than the pre-rebrand version.
- ✅ API: versioned `/v1` prefix with tagged domain routers (`academic`, `planning`, `advisor`, `students`, `admin`); `/v1/plan/generate`, `/v1/plan/edit`, `/v1/advisor/ask`, `/v1/advisor/revise-plan`
- ✅ Direct plan editing: `POST /v1/plan/edit` (`services/plan_editor.py`) — deterministic move/add/remove with placement-preserving re-validation, no LLM
- ✅ **Program/requirement read API already exists and is unused by the UI**: `GET /v1/academic/facets`, `GET /v1/academic/programs` (search/list), `GET /v1/academic/programs/{id}` (full requirement tree). Backend work for Priority 3 is done — only the frontend needs to call it.

**What's missing / rough:**
- 🎯 **No program/degree lookup or picker in the UI** — the `programs` table is fully populated now, but nothing in `clients/web` calls `GET /v1/academic/programs*`. Degree program is still free text and `program_id` is always null. This is the next goal (Priority 3).
- ❌ Graduate programs: not ingested yet (undergrad only, 2026-2027)
- ❌ Colleges / departments: still 0 rows — not populated by the program crawl, no separate ingestion path yet
- ❌ Prerequisite rules: effectively unimplemented — no ingestion pipeline writes real prerequisite data yet (Purdue doesn't publish prereqs in Acalog; needs a separate source)
- ⚠️ **"Edit profile" creates a new student row** — there's no `PUT /v1/students/{id}`, so editing re-creates the student and re-points `localStorage`. Old rows accumulate harmlessly; add an update route to fix properly.
- ⚠️ **The current model choice rests on a retired benchmark.** `VLLM_MODEL` is `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` because the old `model_eval/` said code-tuned pretraining helped "this app's SQL-shaped tasks" — but the app stopped asking a model for SQL when RAG landed (`rag/pipeline.py` retrieves; the model only summarizes). `model_eval/` was rewritten on 2026-07-22 around what the app actually does, with **plan-of-study viability** as the headline metric. The 7B-coder pick is unvalidated until that suite has been run across the bracket: `cd model_eval && python run.py doctor && python run.py run`.
- ⚠️ **The revise-plan proposal is silently no-op-ing for at least one model.** First rewritten-harness run: qwen2.5-7b writes whole semester layouts (`"fall 2026: CS 25000, CS 25100, ... [14 cr]"`) into `PlanEditProposal.reorder`, which expects bare course codes. `advisor_agent._apply_proposal` drops unknown codes by design, so the student gets a plausible rationale and an unchanged plan. Worth considering: tighten the field description, or validate proposals against `remaining_courses` and retry when nothing matched.
- ⚠️ Saved-plan history is write-only from the UI — every save appends a `plans` row, but the UI only ever loads the newest one. No history browser/restore yet (rows are visible in `/admin`).

**Database state** (`make counts` to verify; snapshot 2026-07-07, crawl complete):

| Table | Rows | Status |
|-------|------|--------|
| courses | 10,607 | ✅ Full 2026-2027 import |
| subjects | 208 | ✅ Ready |
| catalog_years | 13 | ✅ All years discovered (2014-2027); only 2026-2027 has programs crawled |
| programs | 1,165 | ✅ **Crawl complete** (2026-2027 undergrad; exceeded the ~959 estimate) |
| requirement_groups | 23,213 | ✅ Filled |
| requirement_options | 40,758 | ✅ Filled |
| colleges | 0 | ❌ Not populated by the crawl |
| departments | 0 | ❌ Not implemented |
| academic_rules (RAG) | 18,603 | ✅ 10,607 course chunks + 7,996 requirement chunks; re-embedded 2026-07-11 at VECTOR(384) (fastembed BAAI/bge-small-en-v1.5), replacing the old Ollama nomic-embed-text VECTOR(768) vectors |

**Impact:** The data a program picker needs (programs + requirement blocks + RAG chunks) is all in place. Program-driven plans and major-specific RAG answers are unblocked at the API layer already (`derive_remaining_courses()` in `planner_catalog.py`) — the only missing piece is UI that actually sets `program_id`. That's Priority 3.

---

## Priority 1: Re-crawl Degree Programs (✅ done 2026-07-06/07)

**Why:** Without programs and requirements, major-specific questions fail. The data was lost when the machine changed. This unblocks many web features.

**Result:** 1,165 undergrad programs (2026-2027, beat the ~959 estimate), 23,213 requirement groups, 40,758 requirement options. Requirement-block RAG chunks backfilled too — `academic_rules` now has 7,996 requirement chunks alongside the 10,606 course chunks. `colleges`/`departments` stayed at 0 (the crawl doesn't populate them; no separate ingestion path exists yet).

```bash
cd catalog_ingestion
docker compose build ingestion   # ensure Playwright image is built (~2 GB)
make sync-programs YEAR=2026-2027 CONCURRENCY=1
make backup                      # save catalog_db_backup.sql.gz when done
```

Requirement-block RAG chunks were then backfilled:
```bash
cd backend && source .venv/bin/activate
python -m app.services.rag.ingest_catalog  # idempotent; adds requirement chunks
```

**Verification:**
```bash
cd catalog_ingestion && make counts  # programs=1165, requirement_groups=23213, requirement_options=40758
```

**Follow-up (not done):** graduate programs, past catalog years (2014-2025 are discovered in `catalog_years` but not crawled), colleges/departments.

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

## Priority 3: Connect the Program Catalog to the Web UI (🎯 NEXT GOAL — fully unblocked)

**Why:** Priority 1 finished: `programs`, `requirement_groups`, and `requirement_options` are fully populated (1,165 programs, 2026-2027), and requirement text is embedded in RAG. But nothing in `clients/web` calls the program endpoints yet — the profile form's degree field is still free text (`program_id` stays `null` forever), and there's no way for a student to just look up "what does a CS major require" without hitting `/docs` directly. This is the concrete next step: let someone search for a degree and see its requirements in the app.

**Scope:** ~4-6 hours. **Backend needs no new work** — it's a read-only integration:
- `GET /v1/academic/facets` → `{catalog_years, schools, subjects}` for building search filters
- `GET /v1/academic/programs?query=&catalog_year=&school=&limit=` → `AcademicProgramSummary[]` (`id`, `school`, `program_title`, `degree_code`, `variant`, `block_count`, `course_count`) for search/autocomplete
- `GET /v1/academic/programs/{id}` → `AcademicProgramDetail` with the full requirement tree: `blocks[]` → `rules[]` (choose-N logic, `raw_text`) → `options[]` → `courses[]` (course code/title/credits)
- `derive_remaining_courses()` (`backend/app/services/planner_catalog.py`) already turns a `program_id` into a real remaining-course list for the planner — it just needs a UI that actually sets `program_id`

**Steps:**
1. **Standalone degree lookup view** — a new page/section (e.g. alongside Planner/Database in `NavBar.tsx`) where anyone can search programs by name/school and expand one to see its full requirement blocks. Pure read, no profile needed: `GET /v1/academic/programs` + `GET /v1/academic/programs/{id}`. This is the literal "pull a degree and its requirements up" feature.
2. **Onboarding picker** — in `ProfileSetup.tsx`, replace the free-text degree field with the same search-and-select UI, and set `program_id` on submit. The field already exists end-to-end (`StudentProfile.program_id` in `backend/app/models/schemas.py`, threaded through `clients/web/lib/api.ts` and `ProfileSetup.tsx`) — it's just never populated from real data today.
3. **Plan generation** — once `program_id` is set, no backend changes are needed; `planner_catalog.py` already derives `remaining_courses` from requirement rows for a real program.
4. (Nice-to-have, feeds Priority 8) Reuse the same requirement-tree rendering from step 1 to show a student's plan progress against their program's requirements.

**Test:** Search "Computer Science," open the BS in Computer Science (2026-2027), see its requirement blocks/rules render; as a new student, pick that same program during onboarding and get a program-specific plan instead of a free-text degree name.

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

**Depends on:** Priority 3 — reuse its requirement-tree rendering rather than building it twice.

**Scope:** ~5 hours.

**Steps:**
1. Fetch `GET /v1/academic/programs/{id}` (returns requirement blocks)
2. Render as collapsible sections: "Core CS" → "Data Structures" (met), "Algorithms" (blocked by prereq)
3. Cross-reference student's completed courses and plan
4. Show which courses satisfy which requirement (why this course matters)

---

## Priority 9: Graduate Programs & Other Years

**Status:** Undergrad 2026-2027 done (Priority 1). Remaining:
- Grad programs (MS/PhD): requires adding navoid to `catalog_ingestion/discover/programs.py`
- Past years (2014-2025 already discovered in `catalog_years`, not crawled): requires `make load-courses YEAR=2025-2026` + `make sync-programs YEAR=2025-2026`

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

- `POST /advisor/ask` retrieves in two tiers: (1) exact — course codes named in the question,
  matched deterministically against `academic_rules.metadata->>'code'`; (2) semantic — the
  question is embedded in-process (fastembed) and the top-K nearest chunks are pulled from
  pgvector. Both tiers are merged and packed under `ADVISOR_CONTEXT_TOKEN_BUDGET` before
  reaching the local llama.cpp prompt (`services/rag/pipeline.py`).
- Returns cited sources so answers are verifiable
- Chunking strategy: one chunk per course + one per requirement block
- Tuning knobs: `MAX_CHUNK_CHARS`, `RAG_TOP_K`, `RAG_MIN_SIMILARITY`, `ADVISOR_CONTEXT_TOKEN_BUDGET` in `.env`

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
- All 10,607 courses embedded in pgvector, plus 7,996 requirement-block chunks (18,602 total)
- Similarity-ranked retrieval working
- Answers cited with sources

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

### 6. Full Degree Program Crawl (✅ Done 2026-07-06/07)
- 1,165 undergrad programs crawled for 2026-2027 (beat the ~959 estimate)
- 23,213 requirement groups, 40,758 requirement options populated
- RAG backfilled with 7,996 requirement-block chunks
- Backend read API (`/v1/academic/facets`, `/v1/academic/programs`, `/v1/academic/programs/{id}`) already built and ready — see Priority 3 for the still-open UI work

### 7. Anthropic API Pivot (✅ Done 2026-07-11, superseded — see item 8)
**No longer the active backend.** The app moved back to a local model (Ollama, then
llama.cpp — item 8) before this file was updated to say so; kept below as history of what
changed at the time, not a description of the current stack.

Moved the advisor from a local Ollama model to the Anthropic API (Claude Haiku). The
deterministic planner/validation core (Priority 2, `services/planner.py`, `plan_editor.py`)
is untouched — this was purely a swap of the LLM/RAG transport layer.
- **Chat:** `services/ollama_client.py` deleted; `services/anthropic_client.py` wraps the
  `anthropic` SDK (`generate()` for free text, `propose()` for schema-enforced structured
  output). The API key is read by the SDK from `ANTHROPIC_API_KEY` — deliberately not a
  `Settings` field, so it can never leak via a `.model_dump()`.
- **`revise-plan` structured outputs:** `PlanEditProposal` (already a Pydantic model) is now
  the Anthropic structured-output schema itself (`client.messages.parse(output_format=...)`);
  the old JSON-repair/retry-on-parse-failure path in `advisor_agent.py` is gone — the API
  guarantees the response validates. The propose→re-plan *semantic* retry loop (a legal
  proposal that made the plan worse) is unchanged.
- **Embeddings:** the Anthropic API has no embeddings endpoint, so `services/rag/embeddings.py`
  now embeds in-process via `fastembed`/ONNX (`BAAI/bge-small-en-v1.5`, 384-d, CPU). This
  required dropping and rebuilding `academic_rules` (old vectors were 768-d from Ollama's
  `nomic-embed-text` and incompatible) and re-running `ingest_catalog` — all chunks
  re-embedded in ~12 minutes on a laptop CPU (18,603 total rows: 10,607 course + 7,996
  requirement chunks — matches the pre-pivot counts within one course chunk). **`catalog_db_backup.sql.gz`
  is now stale on the RAG side** — take a fresh `make backup` before relying on it.
- **Retrieval upgrade:** `services/rag/pipeline.py` now runs exact course-code match
  (`store.fetch_by_course_codes`, SQL, similarity pinned to 1.0) alongside semantic search,
  merges both tiers, and packs them under `ADVISOR_CONTEXT_TOKEN_BUDGET` — input tokens are
  the Anthropic bill, so this cap is enforced in code, not by hope. `RAG_MIN_SIMILARITY`
  raised from `0.0` to `0.45` now that it's actually a meaningful floor.
- **Prompt caching:** `AnthropicClient.system_block()` marks the static system prompt with
  `cache_control` unconditionally — a no-op below Haiku's 4,096-token minimum cacheable
  prefix, a real discount above it.
- **Health check:** `GET /health/llm` replaces `/health/ollama`, using the free Models API
  (`models.retrieve`) so it costs zero tokens.
- **Security:** `is_local_model_endpoint()` (blocked accidental cloud spend from a
  misconfigured local endpoint) is gone — that concern is inverted now that cloud *is* the
  target. Replaced by: no API key in `Settings`, `ANTHROPIC_API_KEY` only ever entering via
  `docker-compose.yml`'s `env_file: ../backend/.env` (gitignored, never inlined, never baked
  into the image), and a workspace-scoped key + Console spend limit (manual setup — see
  `backend/.env.example`).

**Not done as part of this pivot** (deferred, see the original architecture writeup):
streaming responses for `/advisor/ask` (SSE via `client.messages.stream`), and evaluating
Voyage AI as an alternative to in-process embeddings if the VPS CPU turns out to be a
bottleneck at higher traffic.

### 8. llama.cpp Pivot (✅ Done 2026-07-21)
Moved the advisor's local model from Ollama to `llama-server` (llama.cpp's HTTP server),
running `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` — model_eval/'s verdict for the 2060 Super's
8GB budget (best quality among 8GB-class models tried; code-tuned pretraining measurably
helped this app's SQL-shaped tasks). The deterministic planner/validation core is untouched —
purely a swap of the local-inference transport.
- **Chat:** `services/ollama_client.py` deleted; `services/llamacpp_client.py` talks to
  `llama-server`'s OpenAI-compatible `/v1/chat/completions` (default port **8080**, not
  Ollama's 11434). `generate()`/`propose()`/`health()` keep the same surface as before.
- **Structured outputs:** `propose()` now uses `response_format={"type": "json_schema", ...}`,
  which llama.cpp converts to a GBNF grammar server-side — stricter than Ollama's old
  syntax-only `format=<schema>`, but still not a hard guarantee on every schema keyword, so
  the Pydantic-validate-then-retry-once loop stays.
- **Context size / model, no longer per-request:** unlike Ollama, llama-server's context size
  (`--ctx-size`) and loaded model (`--model`) are fixed at process launch — there's no
  `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` equivalent to set per-request. `LLAMACPP_MODEL` in
  `.env` must match the gguf `llama-server` was actually started with; a mismatch is caught by
  `health()` (`/v1/models`), not by the request.
- **Health check:** `GET /health/llm` now hits llama-server's `/health` (reachable?) and
  `/v1/models` (is the loaded model the configured one?), still zero-token.
- **Config:** `Settings`/`.env` renamed `OLLAMA_*` → `LLAMACPP_*`
  (`LLAMACPP_BASE_URL=http://127.0.0.1:8080`, `LLAMACPP_MODEL`, `LLAMACPP_LOCAL_ONLY`,
  `LLAMACPP_TEMPERATURE`, `LLAMACPP_MAX_TOKENS`); `catalog_ingestion/docker-compose.yml`'s
  backend service override moved from `OLLAMA_BASE_URL: http://host.docker.internal:11434` to
  `LLAMACPP_BASE_URL: http://host.docker.internal:8080`.
- **Embeddings unaffected:** RAG embeddings still run in-process via `fastembed`, independent
  of the chat backend — no `academic_rules` re-embed needed for this pivot.

**Not done as part of this pivot:** no launch/systemd script for `llama-server` is checked
into the repo yet (the box currently runs it by hand:
`llama-server -m Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf -ngl 99 --host 0.0.0.0 --port 8080`) —
worth a systemd unit for the 24/7 box, mirroring whatever kept Ollama running as a service.

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
- llama.cpp health check (zero-token — `/health` + `/v1/models`, not inference): `curl http://localhost:8000/health/llm`
- Catalog DB: `postgres://catalog:catalog@localhost:5433/catalog_ingestion`

---

## Known Gotchas

- **Calendar year transitions:** `planner.next_term` correctly crosses year boundaries (fall 2026 → spring 2027)
- **Page cache:** `catalog_ingestion/.page_cache/` speeds up retries — don't delete casually
- **RAG + planner:** Decoupled in code (`rag/store.py` has own connection) — either can be reset independently
- **RAG embedding model is pinned to `academic_rules`'s VECTOR(n) column:** changing `RAG_EMBED_MODEL`/`RAG_EMBED_DIMENSIONS` means the existing table's vectors are no longer comparable to new ones — drop `academic_rules` and re-run `ingest_catalog` (idempotent, ~2 min for 18.6k chunks on CPU). This bit us on the 2026-07-11 Anthropic pivot (Ollama's 768-d `nomic-embed-text` → fastembed's 384-d `bge-small-en-v1.5`).
- **Offline fallback:** `backend/app/data/courses.json` used when DB is down or unknown codes
- **Prerequisite gap:** Purdue doesn't publish prerequisites in Acalog; real prereqs in Banner/myPurduePlan (separate sourcing needed)
- **Fixed (2026-07-05):** `planner.next_term` now correctly crosses year boundaries (fall 2026 → spring 2027)
- **Removed:** `clients/web/lib/mockData.ts` and unused `explainPlan()` client code
- **Tailwind requires its config files:** `clients/web/tailwind.config.ts` + `postcss.config.mjs` are what make Tailwind compile at all. The app shipped for weeks without them and every utility class silently rendered as unstyled HTML. Config changes need a dev-server restart.
- **CSS `@import` must be first:** the Google Fonts import in `globals.css` has to precede the `@tailwind` directives, or browsers drop it after Tailwind expands (fonts silently fall back).
- **Laptop sleep pauses the crawl:** Docker Desktop's VM suspends with the host, so the `sync-programs` container stops fetching while the Mac sleeps and resumes on wake (tenacity retries + the page cache absorb the interrupted request). For the full ~32 h run, prefer an always-on host — the Makefile auto-detects podman vs. Docker, so the stack runs unchanged on a Linux server (`make backup` → copy → `make restore`, then restart `make sync-programs`, ideally copying the page-cache volume too).
