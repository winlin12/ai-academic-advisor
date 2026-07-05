"""Step 2 (ingest) + Step 3 (answer): the composition layer.

This is the only module that imports *both* the embedding transport (OllamaClient) and the
persistence layer (store). Each helper it calls is independently testable; this file just
wires them into the two end-to-end flows.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.services.ollama_client import OllamaClient
from app.services.rag import store

logger = logging.getLogger(__name__)


# Fixed framing for every advisor answer. The retrieved rules are the ONLY facts the model
# is allowed to use, which is what turns a general chatbot into a grounded advisor and stops
# an 8B local model from inventing courses/credits it never saw.
_ADVISOR_SYSTEM_PROMPT = (
    "You are a college academic advisor. Answer using ONLY the CONTEXT below, which contains "
    "retrieved degree rules and course descriptions. If the context does not contain the "
    "answer, say you don't have that rule on file and suggest the student confirm with their "
    "department. Be concise, and quote the specific requirement text you relied on."
)


async def ingest_rule(
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    client: OllamaClient | None = None,
) -> int:
    """Step 2 end-to-end: embed one text chunk, then upsert it into pgvector.

    ``content`` is a self-contained rule/description string (e.g. the STS selective blurb).
    ``metadata`` is arbitrary structured tags (department, requirement_type, ...) stored as
    JSONB for later filtering. Pass a shared ``client`` when bulk-ingesting to reuse config
    and avoid re-running the endpoint guard for every chunk.
    """
    client = client or OllamaClient()
    embedding = await client.embed(content, task_type="search_document")
    rule_id = store.upsert_rule(content, metadata or {}, embedding)
    logger.info("Ingested rule id=%s (%d chars, %d-d vector)", rule_id, len(content), len(embedding))
    return rule_id


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


async def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    """Step 3 end-to-end: embed question -> cosine search -> inject context -> generate.

    Returns the answer plus the matches (with similarities) so callers/UI can show sources
    and so the retrieval quality is observable, not a black box.
    """
    top_k = top_k or settings.rag_top_k
    client = client or OllamaClient()

    # 1-2. Embed the question with the QUERY-side prefix (asymmetric retrieval).
    query_vector = await client.embed(question, task_type="search_query")

    # 3. Nearest chunks, then drop anything below the relevance floor.
    matches = store.search(query_vector, top_k)
    if settings.rag_min_similarity > 0:
        matches = [m for m in matches if (m.get("similarity") or 0.0) >= settings.rag_min_similarity]
    logger.info("Retrieved %d chunk(s) for question: %r", len(matches), question[:80])

    # 4-5. Fold into the prompt and let the chat model write the grounded answer.
    context = _format_context(matches)
    user_prompt = f"CONTEXT:\n{context}\n\nSTUDENT QUESTION:\n{question}"
    answer = await client.generate(_ADVISOR_SYSTEM_PROMPT, user_prompt)

    return {
        "answer": answer,
        "matches": matches,
        "context_char_count": len(context),
        "model": client.model,
    }
