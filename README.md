# AI Academic Advisor

A local-first AI academic planning assistant that helps students reason about degree requirements, prerequisites, semester plans, workload balance, and graduation paths.

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
cd ai-academic-advisor
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
- Shows a plan for one hardcoded demo profile (`demoProfile` in [`clients/web/app/page.tsx`](clients/web/app/page.tsx)) — there is no sign-up/profile-creation form yet, and no major/program picker despite the backend supporting `program_id`
- "Regenerate Plan" reruns the planner from scratch on that same hardcoded profile
- Ask the AI advisor free-text questions (RAG-grounded, cites sources)
- Revise the plan with free-text feedback (“less theory-heavy”, “cap at 6 credits”) — the model proposes reorder/defer/avoid-tag/credit-cap edits, the deterministic planner re-validates them
- **No direct plan editing**: there's no way to move, add, or remove a single course in a specific semester. Regenerate (from scratch) and feedback-driven revise (via the LLM's limited knob set) are the only two ways to change a plan today.
- **No persistence from the UI**: the backend has `POST /students` and `POST /students/{id}/plans` to save profiles/plans, but the web client never calls them — nothing survives a page reload.
- **No database browsing UI**: there's no admin page or table browser. Use `make psql` / `make counts` from a terminal (see [Database Shell](#database-shell)) or point an external SQL client at `localhost:5433`.

**API** (http://localhost:8000/docs):
- `POST /plan/generate` — generate a degree plan
- `POST /advisor/ask` — ask a question with sources
- `POST /advisor/revise-plan` — revise a plan with feedback
- `GET /academic/courses/search` — search the catalog
- `GET /academic/programs/{id}` — view requirements for a major
- `POST /students`, `GET /students/{id}`, `POST /students/{id}/plans` — profile/plan persistence (built, but currently unused by the web client)

⚠️ The API surface is rough right now: routes are a flat, untagged list mixing RPC-style (`/plan/generate`, `/advisor/ask`) and REST-style (`/academic/programs/{id}`, `/students/{id}`) naming with no versioning; `/catalog/courses` is a vestigial endpoint that reads a bundled JSON fixture instead of the real database and duplicates `/academic/courses/search`. See [`TODO.md`](TODO.md) for the cleanup plan.

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
- High-priority next steps (program crawl, direct plan editing, API cleanup, DB browsing, web UI buildout)
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
make backup                    # save to backup.sql.gz
make restore                   # load from backup.sql.gz
```

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
