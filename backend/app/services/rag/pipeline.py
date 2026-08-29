"""Step 2 (ingest) + Step 3 (answer): the composition layer.

This is the only module that composes the embedding provider (``embeddings``), the
persistence layer (``store``), and the chat model (``VllmClient``). Each helper it
calls is independently testable; this file wires them into the two end-to-end flows.

Retrieval is two-tier, cheapest-first:

    1. EXACT — course codes mentioned in the question are answered from SQL directly, with no
       embedding and similarity pinned to 1.0. TWO sources, because they know different
       things: the ``advisor`` schema's prerequisite edges and observed term offerings
       (``planner_db.fetch_course_facts``), and the crawled Acalog description held in
       ``academic_rules``. Acalog publishes no prerequisites at all, so without the first the
       advisor confidently answers "no prerequisites listed" about a course with a four-deep
       chain behind it — while the planner sequences every plan from exactly those edges.
    2. SEMANTIC — the question is embedded and the nearest chunks are pulled by cosine
       distance, floored by ``rag_min_similarity``.

Both tiers are merged (exact first, deduped by row id) and packed under a hard token budget
(``advisor_context_token_budget``) before anything reaches the prompt. Local inference has no
per-token bill, but the cap stays: a bloated context is still slower on an 8GB card and still
a way to accidentally overflow llama-server's fixed ``--ctx-size`` window (unlike Ollama,
there's no per-request num_ctx to silently absorb the overage).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services import planner_db
from app.services.vllm_client import VllmClient
from app.services.rag import embeddings, store

logger = logging.getLogger(__name__)


# The catalog database schema (a committed copy of catalog_ingestion/docs/schema.md; re-sync
# with `cp` when the schema changes). Read once at import so the prompt bytes are identical on
# every request — a per-request read that could pick up a changed file mid-process would
# silently invalidate the prompt cache.
_DB_SCHEMA = (Path(__file__).resolve().parents[2] / "data" / "db_schema.md").read_text(
    encoding="utf-8"
)

# Fixed framing for every advisor answer. The retrieved rules are the ONLY facts the model
# is allowed to use, which is what turns a general chatbot into a grounded advisor and stops
# it from inventing courses/credits it never saw. Static-first ordering (this block, then the
# variable question) is kept even though local llama-server has no explicit prompt-cache API
# like Anthropic's cache_control — it's still good hygiene (keeps a backend swap cheap) and
# lets llama-server's own KV-cache (`--cache-reuse`) reuse an identical prefix across requests.
#
# The schema is appended so the model can interpret the context chunks' metadata tags (they
# mirror these tables/columns) — it contains no course facts itself, never a citable source.
_ADVISOR_SYSTEM_PROMPT = (
    "You are a college academic advisor. Answer using ONLY the CONTEXT below, which contains "
    "retrieved degree rules and course descriptions. If the context does not contain the "
    "answer, say you don't have that rule on file and suggest the student confirm with their "
    "department. Be concise, and quote the specific requirement text you relied on."
    "\n\nBackground: the context chunks are retrieved from a Postgres catalog database with "
    "the schema below. Use it to interpret chunk metadata (table/column names, requirement "
    "types, selective vs. required options) — it contains no course facts itself, so never "
    "cite the schema as a source.\n\n"
    + _DB_SCHEMA
)

# "CS 25100", "cs25100", "MA-16100" — subject prefix + number, tolerant of spacing/hyphen.
_COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,5})[\s-]?(\d{3,5})\b")


async def ingest_rule(content: str, metadata: dict[str, Any] | None = None) -> int:
    """Step 2 end-to-end: embed one text chunk, then upsert it into pgvector.

    ``content`` is a self-contained rule/description string (e.g. the STS selective blurb).
    ``metadata`` is arbitrary structured tags (department, requirement_type, ...) stored as
    JSONB for later filtering.
    """
    embedding = await asyncio.to_thread(embeddings.embed_passage, content)
    rule_id = store.upsert_rule(content, metadata or {}, embedding)
    logger.info("Ingested rule id=%s (%d chars, %d-d vector)", rule_id, len(content), len(embedding))
    return rule_id


# --- pure helpers (no DB, no model — unit-testable) ----------------------------------------


def extract_course_codes(question: str) -> list[str]:
    """Course codes mentioned in a question, normalized to space-free uppercase.

    Exact match beats similarity search every time it applies and costs zero tokens to be
    wrong less often — so codes are extracted deterministically before any embedding runs.
    """
    seen: set[str] = set()
    codes: list[str] = []
    for subject, number in _COURSE_CODE_RE.findall(question):
        code = f"{subject.upper()}{number}"
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _approx_tokens(text: str) -> int:
    # ~4 chars/token is close enough for a packing budget; the free count_tokens endpoint
    # keeps this approximation honest in tests, not in the request path.
    return max(1, len(text) // 4)


def merge_and_budget(
    exact: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    budget_tokens: int,
) -> list[dict[str, Any]]:
    """Merge the two retrieval tiers and pack them under the context token budget.

    Exact matches rank first (they answer the question the student literally asked), then
    semantic matches in similarity order. Duplicates (a code both mentioned and semantically
    near) are dropped by row id. Packing stops at the budget; a chunk that doesn't fit is
    skipped rather than truncated, so every injected chunk stays self-contained.
    """
    merged: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for match in [*exact, *semantic]:
        if match["id"] in seen_ids:
            continue
        seen_ids.add(match["id"])
        merged.append(match)

    packed: list[dict[str, Any]] = []
    remaining = budget_tokens
    for match in merged:
        cost = _approx_tokens(match["content"])
        if cost > remaining:
            continue
        packed.append(match)
        remaining -= cost
    return packed


def _format_context(matches: list[dict[str, Any]]) -> str:
    """Join retrieved chunks into one numbered, tagged block for the prompt.

    Numbering + metadata tags give the model something to cite and keep separate rules from
    bleeding together. Empty match list yields an explicit 'nothing found' marker so the
    system prompt's fallback branch fires instead of the model hallucinating.
    """
    if not matches:
        return "(no relevant catalog rules were found for this question)"
    parts: list[str] = []
    for i, m in enumerate(matches, start=1):
        meta = m.get("metadata") or {}
        tag = " · ".join(f"{k}={v}" for k, v in meta.items())
        similarity = m.get("similarity")
        header = f"[{i}]" + (f" ({tag})" if tag else "") + (f"  ~{similarity:.2f}" if similarity is not None else "")
        parts.append(f"{header}\n{m['content']}")
    return "\n\n".join(parts)


# --- the end-to-end answer flow -------------------------------------------------------------


async def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    client: VllmClient | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Step 3 end-to-end: exact + semantic retrieval -> budgeted context -> generate.

    Returns the answer plus the matches (with similarities) so callers/UI can show sources
    and so the retrieval quality is observable, not a black box.

    ``seed`` only re-rolls the GENERATION. Retrieval is deterministic and deliberately stays
    so: a student pressing "regenerate" on a confusing answer wants it said differently, not
    grounded in different rules, and re-rolling the evidence under them would make two answers
    to the same question disagree for reasons neither of them shows.
    """
    top_k = top_k or settings.rag_top_k
    client = client or VllmClient()

    # Tier 1: deterministic — course codes named in the question, straight from SQL. Two
    # sources, because they know different things: `academic_rules` holds the crawled Acalog
    # description, and the `advisor` schema holds the prerequisites and term offerings Acalog
    # does not publish. Facts first — "what do I need before CS 25100" is the commonest
    # question there is, and answering it from the description alone produces a confident
    # "no prerequisites listed" about a course with a four-deep chain behind it.
    codes = extract_course_codes(question)
    exact = [*planner_db.fetch_course_facts(codes), *store.fetch_by_course_codes(codes)]

    # Tier 2: semantic — embed the question (query-side instruction), cosine search, floor.
    query_vector = await asyncio.to_thread(embeddings.embed_query, question)
    semantic = store.search(query_vector, top_k)
    if settings.rag_min_similarity > 0:
        semantic = [m for m in semantic if (m.get("similarity") or 0.0) >= settings.rag_min_similarity]

    matches = merge_and_budget(exact, semantic, settings.advisor_context_token_budget)
    logger.info(
        "Retrieved %d chunk(s) (%d exact, %d semantic pre-merge) for question: %r",
        len(matches), len(exact), len(semantic), question[:80],
    )

    # Fold into the prompt and let the local model write the grounded answer.
    context = _format_context(matches)
    user_prompt = f"CONTEXT:\n{context}\n\nSTUDENT QUESTION:\n{question}"
    answer = await client.generate(_ADVISOR_SYSTEM_PROMPT, user_prompt, seed=seed)

    return {
        "answer": answer,
        "matches": matches,
        "context_char_count": len(context),
        "model": client.model,
    }
