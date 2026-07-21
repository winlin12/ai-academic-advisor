# BoilerAdvisor

An AI academic planning assistant for Purdue students: reason about degree requirements, prerequisites, semester plans, workload balance, and graduation paths. Catalog data, the deterministic planner, RAG retrieval, and chat reasoning (ask/revise-plan) all run **entirely locally** — Postgres for data, and a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server (`llama-server`) running `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` for the LLM. No cloud API, no per-question spend, no internet required once everything is set up.

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
- **Python 3.11+** — for the backend and ingestion pipeline
- **Node.js 18+** — for the web frontend
- **A GPU with ~8GB VRAM is recommended** (this project was tuned against an RTX 2060 Super) for fast local inference — a CPU-only box also works, just slower. **~5GB free disk** for the model file.
- **llama.cpp** (`llama-server`) — see step 3 below for building/installing it and downloading the model. No API key of any kind is needed for chat.

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

### 3. Start the Local LLM (llama.cpp)

The advisor needs an OpenAI-compatible `llama-server` reachable at `http://127.0.0.1:8080` (the backend's default `LLAMACPP_BASE_URL`) with `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` loaded.

**Get `llama-server`** — pick one:
```bash
# Option A: Homebrew (macOS/Linux)
brew install llama.cpp

# Option B: build from source (needed for CUDA/ROCm GPU support on Linux)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # drop -DGGML_CUDA=ON for a CPU-only build
cmake --build build --config Release -j"$(nproc)"
# binary lands at build/bin/llama-server
```
(See the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases) for prebuilt binaries too.)

**Run it** — `-hf` downloads the model straight from Hugging Face into the standard HF cache the first time (~4.7GB), so no separate download step is needed:
```bash
llama-server -hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M -ngl 99 --port 8080
```
- `-ngl 99` offloads every layer to the GPU (drop it, or lower the number, on a CPU-only box or a smaller GPU).
- No `--ctx-size` flag is needed on an 8GB card — this model's native 32768-token context still fits under 8GB VRAM at Q4_K_M with full GPU offload. Pass `--ctx-size <n>` to cap it lower if you're tighter on VRAM.
- Already have the gguf file downloaded (e.g. from this repo's `models/` dir)? Run `llama-server -m /path/to/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf -ngl 99 --port 8080` instead of `-hf`.

Leave this running — it's a long-lived local service, same role Postgres plays. Verify it's up:
```bash
curl http://127.0.0.1:8080/health   # {"status":"ok"}
```

### 4. Start the Backend API — choose ONE

**Option A: All in Docker (simplest — no Python setup)**

```bash
cd catalog_ingestion
cp ../backend/.env.example ../backend/.env
# Defaults work as-is if llama-server is running on the host per step 3 — no editing required
make up          # starts Postgres + backend containers together
```

The backend container reads `backend/.env` via `env_file` for the RAG tuning knobs, and talks to `llama-server` on the host via `host.docker.internal:8080` (overridden inline in `catalog_ingestion/docker-compose.yml`, since containers can't reach the host via `localhost`). `ACADEMIC_DATABASE_URL` is similarly overridden to point at the `postgres` compose service. Also note: no `--reload`, so backend code edits require `docker compose restart backend`.

**Option B: Native backend (for active development — supports `--reload`)**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Defaults work as-is if llama-server is running on 127.0.0.1:8080 per step 3
# ACADEMIC_DATABASE_URL defaults to localhost:5433, matching the docker-compose port mapping

uvicorn app.main:app --reload --port 8000
```

Either way, visit http://localhost:8000/docs for API docs, and `curl http://localhost:8000/health/llm` to confirm the backend can see `llama-server` and the right model is loaded.

### 5. Start the Web Frontend

```bash
cd clients/web
npm install
npm run dev
```

Open http://localhost:3000

### 6. (Optional) Populate RAG Advisor

The advisor learns from course descriptions via semantic search. Embeddings are produced in-process by `fastembed`/ONNX (`BAAI/bge-small-en-v1.5`, runs on CPU, no separate service or API key needed) — the first embedding call downloads ~100MB of model weights to the local Hugging Face cache. First-time setup:

```bash
cd backend
source .venv/bin/activate

# Preview what will be embedded (no model calls)
python3 -m app.services.rag.ingest_catalog --dry-run

# Embed courses and program requirements (~12 min first time on a laptop CPU)
python3 -m app.services.rag.ingest_catalog
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
- `GET /health`, `GET /health/llm` — liveness / llama-server reachability + loaded-model check (zero-token, hits `/health` + `/v1/models`)
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
| `backend/` | Planner engine, LLM endpoints, RAG store | FastAPI, pgvector, llama.cpp, fastembed |
| `clients/web/` | Student-facing dashboard | Next.js, React, TypeScript |

**Data flow:**
- Purdue Acalog → Postgres (courses, requirements, degree rules)
- Local llama.cpp server (`Qwen2.5-Coder-7B-Instruct-Q4_K_M`, GPU) ← FastAPI → Web UI
- pgvector store (semantic search) ← in-process fastembed embeddings (local, CPU)

## Development Notes

See [`TODO.md`](TODO.md) for:
- Current database state (what's filled, what's missing)
- **Current next goal**: the degree-program crawl is complete (1,165 programs, requirements, RAG chunks), but nothing in the web UI calls the `/v1/academic/programs*` endpoints yet — wiring up a degree/requirements lookup and program picker is the next thing to build (TODO.md Priority 3)
- Technical design decisions and gotchas, including the [llama.cpp Pivot](TODO.md) write-up (moved off Ollama 2026-07-21)

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
Harmless — this is not a database failure. It's caused by something on your machine (commonly an editor's automatic port-forwarding, e.g. VS Code probing newly-opened ports with an HTTP request to see if they're web servers) sending non-Postgres bytes at port 5433. Postgres correctly rejects them and logs it. Ignore it as long as the backend can actually connect (check `curl http://localhost:8000/health/llm` and `GET /academic/*` routes — if those work, the DB connection is fine).

**Backend won't start / "address already in use" on port 8000**
You likely have both Option A (Docker) and Option B (native `uvicorn`) running at once. Check with `lsof -iTCP:8000 -sTCP:LISTEN`, then stop one:
```bash
cd catalog_ingestion && docker compose stop backend   # if using Option A
# or Ctrl-C the native `uvicorn` process if using Option B
```

**Backend connects to the wrong database**
`ACADEMIC_DATABASE_URL` is overridden inline in `catalog_ingestion/docker-compose.yml` for the Docker backend (Option A) so it can reach the `postgres` compose service — that override always wins over whatever's in `backend/.env`. If you need a different DB there, edit `docker-compose.yml` directly, or switch to Option B where `.env`'s `ACADEMIC_DATABASE_URL` is used as-is.

**`curl http://localhost:8000/health/llm` returns `"ok": false`**
`llama-server` isn't reachable, or the loaded model doesn't match `LLAMACPP_MODEL`. Check:
1. Is `llama-server` actually running? `curl http://127.0.0.1:8080/health` should return `{"status":"ok"}`. If not, go back to step 3.
2. Running in Docker (Option A)? The backend container reaches the host via `host.docker.internal:8080` (see `catalog_ingestion/docker-compose.yml`), not `localhost` — make sure `llama-server` is bound to `0.0.0.0` (or at least reachable from Docker), not just `127.0.0.1`, e.g. `llama-server ... --host 0.0.0.0 --port 8080`.
3. Loaded the wrong model? The `/health/llm` detail message names both the configured model and what's actually loaded — restart `llama-server` with `-hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` (or `-m <path-to-the-gguf>`).

**`llama-server` fails to start, or is slow / falls back to CPU**
- Out of VRAM: close other GPU processes, or drop `-ngl 99` to a smaller number to offload fewer layers (the rest run on CPU, slower but still works).
- Built without GPU support: rebuild with `-DGGML_CUDA=ON` (NVIDIA) or the equivalent flag for your GPU vendor — see the [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

**`make sync-programs` / `make db-init` take a long time on first run**
The `ingestion` service builds a ~2GB Playwright image the first time it's invoked. Subsequent runs are fast. Watch for a build step in the output before assuming it's hung.

## Safety & Compliance

- **Fully local**: Catalog data, the planner, the RAG vector store, and the chat model all run on your own machine — nothing about degree requirements, student data, or chat questions is ever sent to a third party. No API key, no internet connection required once Postgres and `llama-server` are running and the model/course data are downloaded.
- **Offline fallback**: Planning/browsing routes work even if `llama-server` is down (uses embedded course data); only `/v1/advisor/*` needs it reachable.
- **Disclaimer**: Always present as a planning tool, not official advising
- **Verification**: Users must confirm major decisions with their university
