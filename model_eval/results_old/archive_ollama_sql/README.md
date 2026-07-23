# Archive — the Ollama-era text-to-SQL harness (retired 2026-07-22)

Kept in version control as history. **Nothing here is runnable** against the current harness,
and none of these results are comparable to new ones.

## Why it was retired

Two independent reasons, either of which alone would have been enough:

1. **The app stopped asking a model for SQL.** When RAG landed, `services/rag/pipeline.py`
   took over retrieval itself — exact course-code lookup plus pgvector cosine search — and the
   model's job shrank to summarizing chunks it was handed. The benchmark was measuring a call
   site that no longer exists, and the ~54 questions and gold-SQL execution-accuracy machinery
   were scoring a capability the product never invokes.

2. **The transport moved to llama.cpp** (2026-07-21). Ollama's `/api/generate`, its nanosecond
   timing counters, its `think:` flag, its per-request `num_ctx`, and `journalctl -u ollama`
   offload parsing have no llama.cpp equivalent — and under llama.cpp, context size is a
   *launch* flag, which changes who has to own the server process.

## What's in here

| File | What it was |
|---|---|
| `questions_sql.yaml` | ~54 text-to-SQL questions with hand-written gold queries |
| `schema.sql` | SQLite stub of the catalog schema + fake seed rows, embedded verbatim in the prompt |
| `ollama_client.py.bak` | Streaming Ollama client (TTFT + Ollama's own token/duration counters) |
| `db.py.bak` | SQLite build, read-only execution, Spider-style execution-accuracy comparison |
| `ollama_pull.sh` | Model-pull script for the two-machine Ollama setup |
| `runs_*.jsonl`, `meta_*.json`, `report.md`, `review_queue*.jsonl`, `manual_grades.md` | Results and grading from the SQL era |

## What replaced it

The plan-of-study eval — see [`../../README.md`](../../README.md). The parts worth carrying
forward were kept: static-first prompts with a hash in every record, the manual review queue
for anything a heuristic can't decide, the refusal to compute a composite score, and the
explicit separation of automatic metrics from ones needing your judgment.

Two ideas from here did **not** carry forward, and it's worth knowing why:

- **Execution accuracy** has no analogue — a plan of study is checked against the fixture's
  rules directly, which is stronger (decidable, no false positives from coincidentally equal
  result sets).
- **The `SQL_CORRECT (attempted)` vs `(honest)` split** existed because abstentions silently
  dropped out of the denominator. The plan metrics have no such escape hatch: a model that
  fails to emit a parseable plan scores `structure_ok = false` and `plan_viable = false`,
  because a plan a student can't read is a plan they didn't get.
