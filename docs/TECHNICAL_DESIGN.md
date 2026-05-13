# Technical Design

## Backend

Framework: FastAPI

Primary responsibilities:

- load catalog
- store/parse student profile
- generate deterministic plans
- call Ollama for explanations
- expose API to mobile/web clients

## Planner Algorithm v0.1

The first planner uses a greedy topological approach.

Input:

- completed courses
- candidate remaining courses
- prerequisites
- term availability
- max credits per semester

Algorithm:

1. Mark completed courses as available.
2. For each term:
   1. Find untaken courses whose prerequisites are satisfied.
   2. Filter by term availability.
   3. Prioritize required courses before electives.
   4. Add courses until max credits reached.
   5. Mark selected courses as completed for future terms.
3. Stop when all requirements are satisfied or no progress can be made.

Limitations:

- does not optimize globally
- does not handle complex “choose N of M” requirements yet
- does not model course conflicts
- does not model professor quality
- does not model exact university registration constraints

## Why Greedy First?

Because v0.1 should be understandable and debuggable.

Later versions can add:

- integer programming
- constraint solving
- search/backtracking
- genetic optimization
- multi-objective scoring

## Ollama Prompting Strategy

The LLM receives structured planner output.

It should be told:

- Do not invent courses.
- Do not change prerequisite decisions.
- Explain only from the supplied plan.
- Mention uncertainty.
- Recommend verifying with official advisor.

## API Design

### GET /health

Basic backend health.

### GET /health/ollama

Checks whether Ollama is reachable.

### GET /catalog/courses

Returns the course catalog.

### POST /plan/generate

Generates a semester plan from a student profile.

### POST /advisor/explain-plan

Uses Ollama to explain a generated plan.

### POST /advisor/ask

Future endpoint for natural language advising.
