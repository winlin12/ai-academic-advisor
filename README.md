# BoilerAdvisor

An AI academic planning assistant for Purdue students: reason about degree requirements, prerequisites, semester plans, workload balance, and graduation paths. Catalog data, the deterministic planner, RAG retrieval, and every LLM call run **entirely locally** — Postgres for data, and a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server (`llama-server`) for the model, launched and managed by the backend itself (`services/model_manager.py`). Three models are available from a picker in the nav bar (Gemma 4 26B by default, plus two Qwen 3.6 variants); only one runs at a time. No cloud API, no per-question spend, no internet required once everything is set up.

**You pick your major and the model writes the plan of study.** It reads the full catalog export for that program — every course, prerequisite chain and observed term offering — and lays out the semesters itself; the deterministic planner then re-checks every placement and repairs anything illegal, so nothing that reaches the screen is a course you cannot register for. See [How it Works](#how-it-works).

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

### 2b. Load prerequisites and term offerings

Purdue's Acalog catalog publishes **neither prerequisites nor term offerings** — the crawl gives
you courses and degree requirements and nothing about sequencing. A planner without that data
believes CS 25100 can be taken in the first semester, so this step is not optional if you want
plans that mean anything.

```bash
cd backend
python -m app.services.seed_catalog     # idempotent; safe to re-run
```

This creates the `advisor` schema (`course_prerequisites`, `course_offering_patterns` +
the `course_planner_terms` view, `course_workload`) and loads it from
[`model_eval/mock_db/`](model_eval/mock_db/), which mirrors those tables row for row. It also
loads the Machine Intelligence program that the eval scores against, so there is a known-good
major to try immediately.

**Know what you are trusting.** Carried over from the fixture's own provenance header:

| | |
|---|---|
| course codes, titles, credits | from the Purdue catalog. Spot-checkable. |
| `offered_terms` | **observed** from PurdueIO Classes rows — a course is "offered in fall" because it *has run* in falls. `terms_observed` is the evidence count; 0 means an editorial guess. |
| prerequisite edges | **hand-written.** Acalog publishes none and Banner's robots.txt disallows crawling. Highest-risk rows in the corpus; stored with a `confidence` column so a reader can see that. |
| `workload_score` | invented 1-5, used only for tie-breaking. |

Coverage is currently ~56 courses. Programs outside that set still plan, with no prerequisite
edges and every course treated as offered every term — legal, but not sequenced. When a real
prerequisite crawl lands it writes the same tables and nothing downstream changes.

### 3. Build `llama-server` and download model files

**The backend launches and stops `llama-server` itself** (`backend/app/services/model_manager.py`) — there is no separate long-lived process to start by hand. You only need to (a) have the binary built and (b) have the gguf files on disk; the backend does the rest at startup, and lets a user switch between three models from a picker in the nav bar without you touching a terminal.

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

**Get the model files** into `models/<name>/` (paths `LLAMACPP_MODELS_ROOT` + each model's `gguf` in `model_manager.AVAILABLE_MODELS` expect):
- `models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` — the default, fastest of the three.
- `models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf` — best plan quality measured, slower.
- `models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` — largest, partly CPU-offloaded.

Set `LLAMACPP_SERVER_EXE` and `LLAMACPP_MODELS_ROOT` in `backend/.env` to point at your binary and models directory (see `.env.example`) — everything else (`--ctx-size`, `--tensor-split`, `--reasoning off`, which flags each model gets) is fixed in code, the same "one launch command, every model identical" discipline `model_eval/harness/server.py` uses, so a model behaves in production exactly as the eval measured it.

**Why these models, and why 26-35B-class at all.** The 7B this originally replaced only ever summarised retrieved chunks; the app now asks the model to emit the schedule itself, which is the task [`model_eval/`](model_eval/) exists to measure and the only one on which models separate sharply. All three are MoE architectures at Q4_K_M — ~16-20GB of weights but only a few billion active parameters per token, so they generate at small-model speed despite the size. On a single 8GB card, none of these three fit well; expect to swap in a smaller gguf and weaker plans — the repair-and-backfill layer still guarantees legality, it just has more to repair.

Once the backend is running (step 4), confirm a model actually loaded:
```bash
curl http://127.0.0.1:8080/health        # {"status":"ok"}
curl http://localhost:8000/v1/models     # {"current": "gemma4-26b", ...}
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
# Defaults work as-is once llama-server is built and the model files are in place per step 3
# ACADEMIC_DATABASE_URL defaults to localhost:5433, matching the docker-compose port mapping

uvicorn app.main:app --reload --port 8000
```

Starting the backend launches the default model automatically (~30-90s before it answers — watch the log for `model_manager: launching`) and stops it cleanly on shutdown. **`--reload` restarts the whole process on every code edit**, which means every edit re-launches the model too — drop `--reload` for a session where you're mostly testing plan/chat behavior rather than editing backend code, to avoid a 30-90s reload after every save.

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
- **Onboarding, no demo profile**: first visit lands on a profile setup form. **Search the real degree catalog and pick your major** — that sets `program_id`, and the backend derives your whole course list from that program's requirement rows (every required course, plus enough options from each selective menu to cover its credits, minus anything in your completed list). Nothing has to be typed course by course. A free-text fallback remains for a program the crawl hasn't reached; that path gets a deterministic plan only, since there is no honest way to write a plan of study for a degree nobody named.
- **The plan is written by the model**, then repaired and backfilled. A provenance panel above the semesters says which — how much of the degree it covers, what was dropped and why, what the planner added — and **Regenerate plan** produces a genuinely different layout (a fresh seed per request, not the same plan redrawn).
- **Plans persist**: creating a profile calls `POST /v1/students`, and every accepted edit/revision/regeneration is autosaved through `POST /v1/students/{id}/plans`. The student id lives in `localStorage`; a reload fetches the newest saved plan from the database. A status chip shows Saved / Saving / Save failed / Not saved (DB down).
- **Chat with three modes** — Catalog (RAG-grounded, cites sources), Explain (about the plan on screen), Revise (“less theory-heavy”, “cap me at 12 credits”). Auto-routed by default with the chosen path shown on every answer; per-message **Regenerate** and **New chat** work the way they do in any chatbot. Only Revise can change your plan.
- **Direct plan editing**: move a course to another semester (per-course dropdown), remove it (×), or add one via a debounced catalog search box on each semester card. Edits go through `POST /v1/plan/edit` — pure deterministic validation (prereqs, term offerings, credit caps), no LLM involved — and warnings/credit totals update instantly.
- **Database browsing** at http://localhost:3000/admin: read-only table browser (row counts + paged row contents) backed by `GET /v1/admin/*`. For full editing, `make adminer` starts [Adminer](https://www.adminer.org/) at http://localhost:8081. `make psql` / `make counts` still work from a terminal (see [Database Shell](#database-shell)).
- **Design**: Purdue-inspired black & gold theme (Boilermaker Gold `#CFB991` on warm black), top navigation between Planner and Database views.

**API** (http://localhost:8000/docs) — all product routes live under `/v1`, grouped into tagged domain routers (`academic`, `planning`, `advisor`, `students`); health probes stay unversioned:
- `GET /health`, `GET /health/llm` — liveness / llama-server reachability + loaded-model check (zero-token, hits `/health` + `/v1/models`)
- `POST /v1/plan/generate` — generate a degree plan (deterministic, milliseconds, no LLM)
- `POST /v1/plan/ai-generate` — **MODE B**: the model writes the plan of study, then it is repaired and backfilled. Tens of seconds. Requires a `program_id`; falls back to the deterministic planner (with `used_model: false`) rather than failing when llama-server is down.
- `POST /v1/plan/requirements` — the degree as a checklist: every requirement group with its slots marked completed/planned/unfilled, and for each hole the courses that could fill it plus the semesters each would legally land in. Deterministic, no LLM.
- `POST /v1/plan/refine` — **MODE C**: `mode` is `fill` (freeze what validates, model fills the gaps), `regenerate` (model is told what's wrong and fixes it, seed fixed) or `start-over` (fresh sample, seed moves). Returns the same envelope as `/ai-generate` plus `kept` and a `note` for the cases where the model wasn't called at all.
- `POST /v1/plan/edit` — move/add/remove one course, deterministically re-validated (no LLM)
- `POST /v1/advisor/ask` — ask a question with sources (RAG)
- `POST /v1/advisor/revise-plan` — **MODE A**: the model turns free-text feedback into a schema-constrained edit proposal, then the plan is rebuilt (`planner: "ai"` re-runs Mode B with it folded in; `"deterministic"` hands it to the greedy planner)
- `POST /v1/advisor/explain-plan` — explain a structured plan
- `GET /v1/academic/courses/search` — search the real catalog
- `GET /v1/academic/facets` — list catalog years/schools/subjects (for building program search filters)
- `GET /v1/academic/programs` — search/list degree programs; backs the onboarding major picker
- `GET /v1/academic/programs/{id}` — full requirement tree for one program
- `POST /v1/students`, `GET /v1/students/{id}`, `POST /v1/students/{id}/plans` — profile/plan persistence (used by the web client for onboarding and plan autosave)
- `GET /v1/admin/tables`, `GET /v1/admin/tables/{table}` — read-only table counts and paged row browsing (whitelisted tables only; backs the `/admin` page)

The old flat, untagged router and the vestigial `/catalog/courses` fixture route are gone (TODO Priority 5): routers now live one-per-domain under [`backend/app/api/routers/`](backend/app/api/routers/), and the bundled fixture survives only as the planner's documented offline fallback.

## How it Works

```
      Major picked in the UI
               ↓
   Postgres: that program's requirement groups, courses,
   prerequisite edges, observed term offerings          ← ~11k-token catalog export
               ↓
   MODE B — the model writes the semesters itself
               ↓
   Checker: prerequisites, corequisites, term offerings, credit caps, major load
               ↓
   Repair — MOVE first, delete only as a last resort
               ↓
   Backfill — the deterministic planner fills what is still unplanned
               ↓
        A plan that is legal by construction, with visible holes
               ↓
   The student fills the holes — by hand from the Requirements view,
   or by asking the model (MODE C: fill / regenerate / start over)
```

**The model proposes, the planner disposes** — but the model now proposes the whole schedule,
not just preferences. Three things keep that safe:

1. **The grammar** constrains what it can emit: terms and course codes are enums built from
   this student's own program, so a hallucinated code is undecodable rather than merely
   discouraged.
2. **The checker** re-derives every hard constraint from the database — the same rules
   `model_eval/harness/plan_scorers.py` scores models on.
3. **Repair moves before it deletes.** A draft that forgot CS 18000 and opened with CS 18200 is
   *one* mistake; a delete-only repair charged it fifteen times and took the whole
   computer-science chain with it. The repairer schedules the missing prerequisite and slides
   the dependents one term later instead. Deletion is the last resort, for the cases nothing
   can save — a course listed twice, a code the catalog has never heard of. **A duplicate keeps
   the EARLIER placement**: anything scheduled after it may depend on it, so dropping the first
   copy would turn one duplicate into a cascade of prerequisite violations.

Whatever it changed is reported, never hidden: the plan view names the courses the model placed,
what was dropped and why, and the one category it may add — a prerequisite the plan forced.

**What it will NOT do is finish the plan for you.** Coverage below 100% is the honest state of a
degree with choices left in it, not a bug.

**Two views of the same plan**, switched by a tab above the semesters:

- **Schedule** — the semester-by-semester grid. Move a course, remove it, add one.
- **Requirements** — the degree as a checklist, myPurduePlan-style, and the view that answers
  *am I going to graduate* (the grid cannot). Every requirement group lists what fills it:

  | | |
  |---|---|
  | **gold ✓** | completed — already on your transcript |
  | **neutral •** | planned — scheduled, not yet taken |
  | **red !** | **not filled** — nothing satisfies it yet |
  | **amber ?** | **can't be checked** — the catalog states it in prose, not as courses |

  Colour is never the only signal: each row also carries a glyph and a word, since roughly one
  man in twelve cannot reliably separate the red from the gold.

  A red slot opens into the courses that could fill it, each with **only the semesters it would
  legally land in** — computed with the same validator the edit route runs, so an option you are
  offered cannot be rejected when you click it. An option whose prerequisites are not scheduled
  yet says *no term available yet* rather than offering a term that would bounce.

**Most requirements can't be checked, and the app says so.** Purdue states a lot of a degree
in prose rather than as a list of courses — *"choose 1 course in 6 different disciplines"*, *"for
a complete listing visit the University Senate Website"*, *"completion of 10100, 10200 in one
world language"*. There is nothing to parse, so those requirements cannot be planned or verified.

They used to be **silently dropped**, which was this app's worst bug: Women's, Gender & Sexuality
Studies showed two requirement groups — both from the major — and a coverage figure of 100%,
while the Liberal Arts core, the university core, the world-language requirement and 20 credits
of electives simply were not on the screen. Across the crawl, **749 of 928 programs** have at
least one; the average program has 3.8 checkable groups and 2.6 uncheckable.

They are now shown in their own *Requires an advisor* section with the catalog's own words, and
kept out of the coverage fraction entirely — which is why it reads "100% of checkable
requirements · 10 unverifiable" rather than a flat, wrong "100%". *Not decidable* is a different
fact from *not met*, and rendering one as the other tells a student they are behind when the
truth is that we don't know.

**Nothing is auto-filled.** The app used to pack leftover courses into free semesters and report
100% coverage. It no longer does, and the plan will legitimately show holes. A selective group is
a MENU — "choose 6 credits from these seventeen" — and taking the first two in catalog order is a
coin toss, not advice; a student who sees CS 44800 in their schedule is entitled to assume
somebody chose it. The only thing still added automatically is a **prerequisite of something the
plan already contains**, which is forced by the plan rather than chosen on your behalf.

**Once you've read the plan**, three buttons above it — MODE C, and they are three different
policies rather than three names for "try again". What separates them is what you KEEP:

| Button | You mean | What happens | Keeps your work? |
|---|---|---|---|
| **Fill the gaps** | "this is close, finish it" | The checker deletes anything illegal, everything that validates is **frozen**, and the model is asked only for what's still missing — with the legal slots for each gap computed and handed to it. | **Yes.** A semester you were happy with cannot be disturbed. |
| **Regenerate** | "this has problems, fix them" | The model is told, specifically, what the checker found and must repair it. Seed held fixed, so a better plan is attributable to the feedback rather than to a luckier sample. | No — it can move anything. |
| **Start over** | "this isn't what I wanted" | A fresh sample, told nothing about the plan it replaces. Seed moves and the temperature creeps, so pressing it again explores instead of redrawing. | No. |

Measured in the browser on a Machine Intelligence plan: 100% covered → **Regenerate** → 67%
(the model traded coverage for the fixes it was told about, which is the feedback variant's
known failure mode) → **Fill** → 89%, *"Kept 21 placement(s) exactly where they were and filled
in 4"* → **Start over** → 100%. That ratchet is the point of Fill.

Fill greys out when there is nothing to fill, rather than spending thirty seconds to tell you
so. And whatever you press, the plan comes back through the same checker — including Fill,
whose merge hands the model a delta it never saw in full.

**In the chat**, three separate backend paths — and only one of them can change your plan:

| You say | Path | What runs |
|---|---|---|
| "what are the prereqs for CS 25100?" | **Catalog** | pgvector retrieval + exact SQL on the `advisor` schema, answer cites its sources |
| "why is CS 38100 so late?" | **Explain** | free text grounded on the structured plan alone |
| "keep me at 12 credits" | **Revise** | MODE A: a schema-constrained edit proposal, then the plan is rebuilt and re-validated |

The mode is auto-detected and shown on every answer; a segmented control overrides it. Every
answer has its own **Regenerate** (a revision rewinds to the plan it started from first, so
pressing it twice does not apply the edit twice), and **New chat** clears the conversation
without touching the plan.

## Architecture

| Component | Role | Technology |
|-----------|------|-----------|
| `catalog_ingestion/` | Scrape Purdue catalog, populate database | Python, Postgres, Playwright |
| `backend/` | Planner engine, LLM endpoints, RAG store | FastAPI, pgvector, llama.cpp, fastembed |
| `clients/web/` | Student-facing dashboard | Next.js, React, TypeScript |

**Data flow:**
- Purdue Acalog → Postgres (courses, requirements, degree rules)
- Local llama.cpp server (one of three models, GPU, launched/switched by FastAPI itself — `services/model_manager.py`) ← FastAPI → Web UI
- pgvector store (semantic search) ← in-process fastembed embeddings (local, CPU)

## Development Notes

See [`TODO.md`](TODO.md) for:
- Current database state (what's filled, what's missing)
- **Where the data thins out**: the program picker, program-driven course derivation and the AI planner are all live. What is still narrow is `advisor.course_prerequisites` / `course_offering_patterns` — ~56 courses, loaded from the eval fixture by `seed_catalog` (see [step 2b](#2b-load-prerequisites-and-term-offerings)). Any program whose courses fall outside that set plans without prerequisite edges: legal, but not sequenced. A real Banner prerequisite crawl is the highest-value next thing to build, and it writes the same tables.
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
`llama-server` isn't reachable, or the loaded model doesn't match what `model_manager.py` thinks it launched. Check:
1. Did it launch at all? `curl http://localhost:8000/v1/models` — `last_error` names why, if the startup launch failed (bad `LLAMACPP_SERVER_EXE`/`LLAMACPP_MODELS_ROOT` path, out of VRAM, port already bound). The backend logs `model_manager: launching <name>` on every attempt.
2. Running in Docker (Option A)? **The auto-launch does not work there** — a container can't spawn a process on the host, and `LLAMACPP_SERVER_EXE`/`LLAMACPP_MODELS_ROOT` point at host paths that don't exist inside it. `start_default()` fails best-effort (the app still starts; catalog browsing and the deterministic planner don't need an LLM at all) and the model picker will show "No model running". For Option A, run `llama-server` on the host by hand as before (see this section's git history for the old manual-launch flags) bound to `0.0.0.0` so `host.docker.internal:8080` can reach it — the picker's switch button won't work against a hand-launched server, only reachability will.
3. Loaded the wrong model? The `/health/llm` detail message names both the configured model and what's actually loaded. Switch it from the nav bar's model picker, or `POST /v1/models/<name>/select` directly (`<name>` is one of `services/model_manager.AVAILABLE_MODELS`' `name` fields, e.g. `gemma4-26b`).

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
