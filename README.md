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
  - Ollama running on GPU PC
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
```

Use your PC's Tailscale IP.

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
