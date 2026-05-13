# Detailed Project Plan: AI Academic Advisor

## 1. Product Vision

Build a personal AI academic advisor that helps students plan their degree path, understand requirements, avoid prerequisite traps, and create semester-by-semester schedules.

The app should feel like a smart planning assistant that understands degree audits, course catalogs, prerequisites, student preferences, and academic risk.

It should not claim to replace official advising. Its role is to help the student think clearly before talking to an advisor.

## 2. Core User Pain Points

### Pain Point 1: Degree requirements are hard to reason about

Students often see a degree audit but do not know:

- which requirements are truly blocking graduation
- which classes double-count
- which requirements are flexible
- which course choices create future bottlenecks

### Pain Point 2: Prerequisite chains are easy to miss

A course may look available, but actually depends on a chain of prior classes.

```text
Course C requires Course B
Course B requires Course A
Course A is only offered in Fall
```

This can delay graduation by a full year.

### Pain Point 3: Semester workload is not just credit hours

A 12-credit semester with compilers, operating systems, and algorithms may be harder than an 18-credit semester with lighter electives.

The app should eventually support workload tags:

- theory-heavy
- project-heavy
- exam-heavy
- coding-heavy
- reading-heavy
- group-project-heavy

### Pain Point 4: Students want what-if answers

Examples:

- What if I fail this class?
- What if I drop this semester to 9 credits?
- What if I work 20 hours/week?
- What if I only want project-heavy courses?
- What if I want to graduate one semester earlier?
- What if this class is not offered next spring?

## 3. Non-Goals for v1

Avoid these at first:

- Full production authentication
- Multi-university support
- Perfect PDF parsing
- Official integration with university systems
- Complex UI polish
- Fine-tuning models
- Automatic enrollment
- Guaranteed correctness claims

The first version should be a reliable prototype.

## 4. System Architecture

```text
+-------------------+
|  Client Layer     |
|-------------------|
| SwiftUI iOS app   |
| Web dashboard     |
| CLI/dev scripts   |
+---------+---------+
          |
          v
+-------------------+
| FastAPI Backend   |
|-------------------|
| Student profile   |
| Course catalog    |
| Requirement model |
| Planner engine    |
| LLM explanations  |
+---------+---------+
          |
          v
+-------------------+
| Local AI Layer    |
|-------------------|
| Ollama on PC GPU  |
| Coding/chat model |
+-------------------+
```

## 5. Planning Philosophy

The key design decision:

> The deterministic planner decides. The LLM explains.

Do not let the LLM be the source of truth for prerequisite correctness.

The planner should handle:

- completed courses
- course availability
- prerequisites
- credit limits
- requirement categories
- required courses
- elective slots
- impossible plans

The LLM should handle:

- natural language explanation
- tradeoff discussion
- friendly summaries
- user-facing advice
- why this plan works or does not work

## 6. Data Model

### Course

Fields:

- code
- title
- credits
- prereqs
- offered terms
- requirement tags
- workload score
- notes

Example:

```json
{
  "code": "CS502",
  "title": "Compilers",
  "credits": 3,
  "prereqs": ["CS251"],
  "offered_terms": ["spring"],
  "requirement_tags": ["systems", "project-heavy"],
  "workload_score": 5
}
```

### Student Profile

Fields:

- student name
- degree program
- completed courses
- target graduation term
- max credits per semester
- preferences
- remaining requirements

### Semester Plan

Fields:

- term
- year
- courses
- total credits
- workload estimate
- warnings
- explanation

## 7. Version Roadmap

### v0.1 — Backend Skeleton

Goal: Make the backend run and generate a simple plan from JSON.

Features:

- FastAPI app
- Course catalog loader
- Student profile parser
- Basic topological prerequisite planner
- Ollama health check
- LLM explanation endpoint

Success criteria:

- `/health` works
- `/catalog/courses` returns seed courses
- `/plan/generate` returns a semester plan
- impossible prerequisite cases return clear warnings

### v0.2 — Better Planner

Goal: Make planning more realistic.

Features:

- fall/spring/summer course availability
- max credits per semester
- completed course handling
- workload balancing
- required vs elective categories
- simple what-if scenarios

Success criteria:

- App can explain why a course cannot be taken yet
- App can detect when graduation target is impossible
- App can generate at least two alternate plans

### v0.3 — Natural Language Advisor

Goal: Let users ask questions conversationally.

Features:

- `/advisor/ask`
- converts user question into planner calls
- uses LLM to summarize planner output
- keeps deterministic planner as source of truth

Example questions:

- Can I take CS502 next semester?
- What if I drop CS541?
- Why can’t I graduate by Spring 2027?
- Give me the safest plan.

### v0.4 — PDF Import

Goal: Ingest degree audits and course catalogs.

Features:

- upload PDF
- extract text
- identify completed courses
- identify requirement blocks
- manual correction screen later

PDF import should not be trusted blindly. Always show extracted data to the user for confirmation.

### v0.5 — Web UI

Goal: Make a simple dashboard.

Suggested stack:

- Next.js or Vite
- course catalog page
- profile editor
- generated semester plan
- advisor chat panel

### v0.6 — SwiftUI App

Goal: Make it feel like a mobile assistant.

Features:

- profile setup
- semester plan cards
- advisor chat
- risk warnings
- what-if buttons

### v0.7 — RAG

Goal: Ground answers in course catalog and requirement documents.

Features:

- vector store
- chunked course descriptions
- degree requirement documents
- citation-style source snippets

### v1.0 — Personal Academic Planning Assistant

Goal: A usable local-first app.

Features:

- stable planner
- polished UI
- import/export plans
- advisor disclaimers
- local model support
- Tailscale/private server support

## 8. Milestone Checklist

### Milestone A: Remote Dev Ready

- [ ] Mac can SSH into PC
- [ ] VS Code Remote SSH works
- [ ] Continue.dev can call PC Ollama
- [ ] Backend repo opens remotely
- [ ] `curl http://PC_TAILSCALE_IP:11434/api/tags` works

### Milestone B: Backend v0.1

- [ ] FastAPI runs
- [ ] Catalog loads
- [ ] Student profile loads
- [ ] Planner generates plan
- [ ] Planner returns warnings
- [ ] Unit tests pass

### Milestone C: AI Advisor v0.1

- [ ] Ollama health endpoint works
- [ ] LLM can summarize plan
- [ ] LLM is prevented from inventing courses
- [ ] Prompt includes planner output as source of truth

### Milestone D: UI Prototype

- [ ] Web or SwiftUI client calls backend
- [ ] Displays semester plan
- [ ] Displays warnings
- [ ] Provides simple chat input

## 9. Engineering Rules

1. Planner correctness beats LLM cleverness.
2. Every course recommendation must come from catalog data.
3. Every prerequisite decision must be explainable.
4. Every imported PDF field must be reviewable.
5. The app must clearly disclaim that it is not official advising.
6. Keep v1 local-first and private.
7. Avoid overbuilding auth until the core app is useful.
