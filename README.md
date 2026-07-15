# BoilerAdvisor

A local-first AI academic planning assistant for Purdue students: reason about degree requirements, prerequisites, semester plans, workload balance, and graduation paths.

## What it does

Ask questions like:
- “Can I graduate in 3 semesters?”
- “What should a CS major take first?”
- “Which classes should I take next semester?”
- “Why is this plan impossible?”
- “Can you make my final year less theory-heavy?”

⚠️ **This is a planning assistant, not an official academic advisor.** Users should verify important decisions with their university.

## Getting Started

### Prerequisites

- **macOS or Linux**
- **Docker Desktop** (or podman) — required for the Postgres database
- **Python 3.9+** — for the backend and ingestion pipeline
- **Node.js 18+** — for the web frontend
- **Ollama** — download from [ollama.ai](https://ollama.ai)

### 1. Clone & Install

```bash
git clone <repo>
cd ai-academic-advisor   # repo directory name predates the BoilerAdvisor rename
```

There are two ways to run Postgres + the backend API: all in Docker (simplest, no Python setup) or the backend natively (needed for live-reload while editing backend code). **Pick one — don't run both at once, they'll fight over port 8000.**

### 2. Start the Database

```bash
cd catalog_ingestion
docker compose up -d postgres     # starts Postgres on port 5433
```

First time? Create schema and import data:
```bash
make db-init                   # creates tables, indexes (auto-builds the ingestion image first time, ~2GB, a few minutes)
make load-courses YEAR=2026-2027  # imports course catalog
```

Restore from backup instead (recommended, includes RAG vectors):
```bash
make restore                   # from catalog_db_backup.sql.gz
```

### 3. Start the Backend API — choose ONE

**Option A: All in Docker (simplest — no Python setup)**

```bash
cd catalog_ingestion
make up          # starts Postgres + backend containers together
```

⚠️ This backend container does **not** read `backend/.env` — its DB and Ollama URLs are hardcoded in `catalog_ingestion/docker-compose.yml` (`ACADEMIC_DATABASE_URL` points at the `postgres` service, `OLLAMA_BASE_URL` points at `host.containers.internal:11434`, i.e. Ollama running on your host machine). Edit that file directly if you need different values. Also note: no `--reload`, so backend code edits require `docker compose restart backend`.

**Option B: Native backend (for active development — supports `--reload`)**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure Ollama endpoint + DB connection
cp .env.example .env
# Edit .env: set OLLAMA_BASE_URL to your Ollama server (default http://localhost:11434)
# ACADEMIC_DATABASE_URL defaults to localhost:5433, matching the docker-compose port mapping

uvicorn app.main:app --reload --port 8000
```

Either way, visit http://localhost:8000/docs for API docs.

### 4. Start the Web Frontend

```bash
cd clients/web
npm install
npm run dev
```

Open http://localhost:3000

### 5. (Optional) Populate RAG Advisor

The advisor learns from course descriptions via semantic search. First-time setup:

```bash
cd backend
source .venv/bin/activate

# Ensure embedding model is downloaded
ollama pull nomic-embed-text

# Preview what will be embedded (no model calls)
python -m app.services.rag.ingest_catalog --dry-run

# Embed courses and program requirements (~75 min first time)
python -m app.services.rag.ingest_catalog
```

## Using the App

**Web UI** (http://localhost:3000) — current state, not aspirational:
- **Onboarding, no demo profile**: first visit lands on a profile setup form (name, degree, start term/year, credit cap, completed courses + courses to schedule via catalog search). There is no hardcoded default student. A major/program *picker* is still pending — the `programs` table is now fully populated (1,165 undergrad programs, 2026-2027; see [`TODO.md`](TODO.md) Priority 1), but the UI hasn't been wired up to it yet, so degree name is still free text (Priority 3, the current next goal).
- **Plans persist**: creating a profile calls `POST /v1/students`, and every accepted edit/revision/regeneration is autosaved through `POST /v1/students/{id}/plans`. The student id lives in `localStorage`; a reload fetches the newest saved plan from the database. A status chip shows Saved / Saving / Save failed / Not saved (DB down).
- Ask the AI advisor free-text questions (RAG-grounded, cites sources)
- Revise the plan with free-text feedback (“less theory-heavy”, “cap at 6 credits”) — the model proposes reorder/defer/avoid-tag/credit-cap edits, the deterministic planner re-validates them
- **Direct plan editing**: move a course to another semester (per-course dropdown), remove it (×), or add one via a debounced catalog search box on each semester card. Edits go through `POST /v1/plan/edit` — pure deterministic validation (prereqs, term offerings, credit caps), no LLM involved — and warnings/credit totals update instantly.
- **Database browsing** at http://localhost:3000/admin: read-only table browser (row counts + paged row contents) backed by `GET /v1/admin/*`. For full editing, `make adminer` starts [Adminer](https://www.adminer.org/) at http://localhost:8081. `make psql` / `make counts` still work from a terminal (see [Database Shell](#database-shell)).
- **Design**: Purdue-inspired black & gold theme (Boilermaker Gold `#CFB991` on warm black), top navigation between Planner and Database views.

**API** (http://localhost:8000/docs) — all product routes live under `/v1`, grouped into tagged domain routers (`academic`, `planning`, `advisor`, `students`); health probes stay unversioned:
- `GET /health`, `GET /health/ollama` — liveness / local-model status
- `POST /v1/plan/generate` — generate a degree plan (deterministic)
- `POST /v1/plan/edit` — move/add/remove one course, deterministically re-validated (no LLM)
- `POST /v1/advisor/ask` — ask a question with sources (RAG)
- `POST /v1/advisor/revise-plan` — revise a plan with free-text feedback (LLM proposes, planner disposes)
- `POST /v1/advisor/explain-plan` — explain a structured plan
- `GET /v1/academic/courses/search` — search the real catalog
- `GET /v1/academic/facets` — list catalog years/schools/subjects (for building program search filters)
- `GET /v1/academic/programs` — search/list degree programs (1,165 loaded for 2026-2027); **not yet called from the web UI** (TODO Priority 3)
- `GET /v1/academic/programs/{id}` — full requirement tree for one program; **not yet called from the web UI** (TODO Priority 3)
- `POST /v1/students`, `GET /v1/students/{id}`, `POST /v1/students/{id}/plans` — profile/plan persistence (used by the web client for onboarding and plan autosave)
- `GET /v1/admin/tables`, `GET /v1/admin/tables/{table}` — read-only table counts and paged row browsing (whitelisted tables only; backs the `/admin` page)

The old flat, untagged router and the vestigial `/catalog/courses` fixture route are gone (TODO Priority 5): routers now live one-per-domain under [`backend/app/api/routers/`](backend/app/api/routers/), and the bundled fixture survives only as the planner's documented offline fallback.

## How it Works

```
Student Profile (major, completed courses, constraints)
         ↓
   Database (course catalog, degree requirements)
         ↓
 Deterministic Planner (validates prerequisites, term offerings, credit caps)
         ↓
    Valid Schedule ← AI refines with feedback
         ↓
  Citable Answers (RAG retrieves from course catalog)
```

The AI proposes, the planner validates. The planner is the source of truth — no hallucinated courses or broken prerequisites.

## Architecture

| Component | Role | Technology |
|-----------|------|-----------|
| `catalog_ingestion/` | Scrape Purdue catalog, populate database | Python, Postgres, Playwright |
| `backend/` | Planner engine, LLM endpoints, RAG store | FastAPI, pgvector, Ollama |
| `clients/web/` | Student-facing dashboard | Next.js, React, TypeScript |

**Data flow:**
- Purdue Acalog → Postgres (courses, requirements, degree rules)
- Ollama (local LLM) ← FastAPI → Web UI
- pgvector store (semantic search) ← Ollama embeddings

## Development Notes

See [`TODO.md`](TODO.md) for:
- Current database state (what's filled, what's missing)
- **Current next goal**: the degree-program crawl is complete (1,165 programs, requirements, RAG chunks), but nothing in the web UI calls the `/v1/academic/programs*` endpoints yet — wiring up a degree/requirements lookup and program picker is the next thing to build (TODO.md Priority 3)
- Technical design decisions and gotchas

### Testing

```bash
cd backend
pytest                         # run all tests
pytest -xvs tests/test_planner.py   # run specific tests
```

### Database Shell

```bash
cd catalog_ingestion
make psql                      # SQL prompt
make counts                    # row counts per table
make adminer                   # browser DB client at http://localhost:8081
make backup                    # save to backup.sql.gz
make restore                   # load from backup.sql.gz
```

For read-only browsing without leaving the app, use http://localhost:3000/admin.

## Troubleshooting

**`invalid length of startup packet` in Postgres logs (`docker logs catalog-ingestion-postgres`)**
Harmless — this is not a database failure. It's caused by something on your machine (commonly an editor's automatic port-forwarding, e.g. VS Code probing newly-opened ports with an HTTP request to see if they're web servers) sending non-Postgres bytes at port 5433. Postgres correctly rejects them and logs it. Ignore it as long as the backend can actually connect (check `curl http://localhost:8000/health/ollama` and `GET /academic/*` routes — if those work, the DB connection is fine).

**Backend won't start / "address already in use" on port 8000**
You likely have both Option A (Docker) and Option B (native `uvicorn`) running at once. Check with `lsof -iTCP:8000 -sTCP:LISTEN`, then stop one:
```bash
cd catalog_ingestion && docker compose stop backend   # if using Option A
# or Ctrl-C the native `uvicorn` process if using Option B
```

**Backend connects to the wrong database / edits to `backend/.env` have no effect**
That means you're running Option A (Docker). The Docker backend container ignores `backend/.env`; its connection settings are hardcoded in `catalog_ingestion/docker-compose.yml`. Edit that file, or switch to Option B for `.env`-driven config.

**`ollama` calls fail from the Docker backend but work natively**
The Docker backend reaches your host's Ollama via `host.containers.internal`, mapped to the host gateway in `docker-compose.yml`. Make sure Ollama is bound to an interface reachable from containers (`OLLAMA_HOST=0.0.0.0 ollama serve`), not just `127.0.0.1`.

**`make sync-programs` / `make db-init` take a long time on first run**
The `ingestion` service builds a ~2GB Playwright image the first time it's invoked. Subsequent runs are fast. Watch for a build step in the output before assuming it's hung.

## Safety & Compliance

- **Local-first**: No data sent to cloud LLM providers (Ollama only)
- **Offline fallback**: App works without internet (uses embedded course data)
- **Disclaimer**: Always present as a planning tool, not official advising
- **Verification**: Users must confirm major decisions with their university
