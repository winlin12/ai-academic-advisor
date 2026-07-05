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

### 2. Start the Database

```bash
cd catalog_ingestion
docker compose up postgres     # starts Postgres on port 5433
```

First time? Create schema and import data:
```bash
make db-init                   # creates tables, indexes
make load-courses YEAR=2026-2027  # imports course catalog
```

Restore from backup (recommended, includes RAG vectors):
```bash
make restore                   # from catalog_db_backup.sql.gz
```

### 3. Start the Backend API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e “.[dev]”

# Configure Ollama endpoint
cp .env.example .env
# Edit .env: set OLLAMA_BASE_URL to your Ollama server
# Default is http://localhost:11434 (Ollama running locally)

uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for API docs.

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

**Web UI** (http://localhost:3000):
- Create a student profile (major, completed courses, start term)
- Generate a degree plan for your remaining semesters
- Ask the AI advisor free-text questions
- Revise plans with natural language feedback (“less theory-heavy”, “6 credits/term”)

**API** (http://localhost:8000/docs):
- `POST /plan/generate` — generate a degree plan
- `POST /advisor/ask` — ask a question with sources
- `POST /advisor/revise-plan` — revise a plan with feedback
- `GET /academic/courses/search` — search the catalog
- `GET /academic/programs/{id}` — view requirements for a major

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
- High-priority next steps (program crawl, web UI buildout)
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

## Safety & Compliance

- **Local-first**: No data sent to cloud LLM providers (Ollama only)
- **Offline fallback**: App works without internet (uses embedded course data)
- **Disclaimer**: Always present as a planning tool, not official advising
- **Verification**: Users must confirm major decisions with their university
