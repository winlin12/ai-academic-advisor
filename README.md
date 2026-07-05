# AI Academic Advisor

A local-first AI academic planning assistant that helps students reason about degree requirements, prerequisites, semester plans, workload balance, and graduation paths.

## Project Goal

Build a mobile/web-friendly AI academic advisor that can answer questions like:

- “Can I graduate in 3 semesters?”
- “What happens if I fail CS 240?”
- “Which classes should I take next semester?”
- “Why is this plan impossible?”
- “Can you make my final year less theory-heavy?”
- “What prereqs block me from taking this class?”
- “What is the safest schedule if I work 20 hours/week?”

This is intended as a fun technical project, not an official academic advising replacement.

## Architecture

```text
Client Layer
  - SwiftUI iOS app later
  - Web dashboard later
  - CLI/dev scripts now

Backend Layer
  - FastAPI
  - Degree planner engine
  - Course/prereq graph
  - Ollama LLM client
  - SQLite later

AI Layer
  - Local/self-hosted Ollama-compatible model endpoint
  - Qwen Coder / Llama / Gemma-style local models
  - RAG over course catalog and degree rules later

Data Layer
  - Seed course catalog JSON
  - Requirement templates
  - Student profile JSON
```

## Quick Start

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
OLLAMA_BASE_URL=http://100.x.y.z:11434
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_LOCAL_ONLY=true
```

Use your Mac, LAN machine, or GPU PC's Tailscale IP. The backend defaults to
local-only model endpoints and rejects public cloud LLM hosts so test calls do
not accidentally spend API money. Local models can use substantial CPU/GPU,
memory, power, and battery; students should opt into that knowingly.

Install and run:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Test Ollama:

```bash
curl http://localhost:8000/health/ollama
```

Generate a starter plan:

```bash
curl -X POST http://localhost:8000/plan/generate \
  -H "Content-Type: application/json" \
  -d @app/data/sample_student_profile.json
```

## Advisor (RAG)

The advisor at `POST /advisor/ask` answers free-text questions by semantic retrieval: it
embeds the question, pulls the most similar catalog chunks from the pgvector `academic_rules`
table, and grounds the local model on those. Retrieved chunks are returned as `sources` so
answers stay citable.

That table starts empty, so populate it once from the ingested catalog (re-run when the
catalog changes; it's idempotent):

```bash
ollama pull nomic-embed-text            # embedding model (separate from the chat model)
python -m app.services.rag.ingest_catalog --dry-run   # preview chunks, no model calls
python -m app.services.rag.ingest_catalog             # embed courses + program rules
```

Then ask:

```bash
curl -X POST http://localhost:8000/advisor/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "I want to major in CS — what should I take first?"}'
```

## Current Scope

Version 0.1 focuses on deterministic planning first:

- Course catalog
- Prerequisites
- Completed courses
- Remaining requirements
- Semester capacity
- Simple workload balancing
- Basic impossibility detection

The LLM is initially used for explanation, not for making the core planning decision.

## Suggested Development Order

1. Make the backend run.
2. Make planner work on seed data.
3. Add more realistic course/requirement modeling.
4. Add PDF import for degree audits.
5. Add RAG over course catalog.
6. Add SwiftUI or web client.
7. Add calendar/registration reminders.
8. Add multi-school support.

## Safety / Disclaimer

This app should always present itself as a planning assistant, not an official academic advisor. Users should verify important decisions with their university.
