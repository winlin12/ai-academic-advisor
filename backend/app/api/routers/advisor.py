"""LLM-facing routes: ask (RAG), explain-plan, revise-plan. These are the only routes that
call the local llama.cpp model; everything they return that *matters* (the plan itself) is
still produced or re-validated by the deterministic planner."""

import psycopg
from fastapi import APIRouter, HTTPException

from app.api.deps import (
    academic_db_unavailable,
    resolve_for_ai_planning,
    resolve_for_planning,
)
from app.models.schemas import (
    AdvisorAskRequest,
    AdvisorAskResponse,
    AdvisorSource,
    ExplainPlanRequest,
    ExplainPlanResponse,
    RevisePlanRequest,
    RevisePlanResponse,
)
from app.services.advisor_agent import ai_revise_plan, revise_plan
from app.services.llamacpp_client import (
    LlamaCppClient,
    LlamaCppConnectionError,
    ModelResponseError,
)
from app.services.rag.embeddings import EmbeddingError
from app.services.rag.pipeline import answer_question

router = APIRouter(prefix="/advisor", tags=["advisor"])

_LlmError = LlamaCppConnectionError | ModelResponseError


def _llm_http_error(exc: _LlmError) -> HTTPException:
    """Map the local-model failure modes onto the HTTP ladder, most specific first.

    503 = the model backend isn't usable right now (llama-server down) — retryable once an
    operator fixes it, not by the client retrying immediately. 502 = the server answered but
    the output was unusable (empty, non-2xx, or still-invalid JSON after propose()'s internal
    retry). There is no upstream rate limit to map (no cloud vendor).
    """
    if isinstance(exc, LlamaCppConnectionError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@router.post("/explain-plan", response_model=ExplainPlanResponse)
async def explain_plan(req: ExplainPlanRequest):
    client = LlamaCppClient()

    system_prompt = """
You are an AI academic planning assistant.
You are not an official academic advisor.
You must not invent courses, prerequisites, requirements, or policies.
Explain only from the supplied structured plan.
Always recommend verifying important decisions with an official advisor.
Be concise unless the student asks for depth.
""".strip()

    user_prompt = f"""
Student question:
{req.question}

Structured plan:
{req.plan.model_dump_json(indent=2)}

Explain the plan clearly. Focus on prerequisites, semester sequencing, warnings, and risk.
""".strip()

    try:
        answer = await client.generate(
            system_prompt=system_prompt, user_prompt=user_prompt, seed=req.seed
        )
    except (LlamaCppConnectionError, ModelResponseError) as exc:
        raise _llm_http_error(exc) from exc

    return ExplainPlanResponse(answer=answer)


@router.post("/revise-plan", response_model=RevisePlanResponse)
async def revise_plan_route(req: RevisePlanRequest):
    """Revise a plan from free-text feedback via the structured-output agent (MODE A).

    The model proposes edits (reorder/defer/avoid-tags/credit-cap) as a schema-enforced
    PlanEditProposal, so a hallucinated proposal degrades to a no-op instead of an illegal
    plan. What happens next depends on ``planner``:

      "ai"            re-runs Mode B with the proposal folded in, so the revision looks like
                      the plan it revised. Requires a ``program_id``.
      "deterministic" hands the proposal to the greedy planner — no GPU, milliseconds.

    A profile with no program falls back to "deterministic" regardless of what was asked for,
    since Mode B has no catalog to plan from without one. Error ladder: 503 (llama-server
    down), 502 (bad output).
    """
    client = LlamaCppClient()

    if req.planner == "ai" and req.profile.program_id:
        profile, program_catalog = resolve_for_ai_planning(req.profile)
        try:
            return await ai_revise_plan(
                profile, program_catalog, req.feedback, req.current_plan,
                client=client, seed=req.seed,
            )
        except (LlamaCppConnectionError, ModelResponseError) as exc:
            raise _llm_http_error(exc) from exc

    profile, catalog = resolve_for_planning(req.profile)
    try:
        return await revise_plan(profile, catalog, req.feedback, client=client, seed=req.seed)
    except (LlamaCppConnectionError, ModelResponseError) as exc:
        raise _llm_http_error(exc) from exc


@router.post("/ask", response_model=AdvisorAskResponse)
async def advisor_ask(req: AdvisorAskRequest):
    """Answer a free-text student question via exact + semantic (pgvector) retrieval.

    Course codes named in the question are fetched deterministically; the question is also
    embedded (in-process) and the most similar ``academic_rules`` chunks are pulled by
    cosine distance. The local model is grounded on just those, under a hard context budget.
    The retrieved chunks come back as ``sources`` so the answer stays inspectable and
    citable. The catalog must first be loaded into ``academic_rules``
    (``python -m app.services.rag.ingest_catalog``), otherwise retrieval returns nothing
    and the model will say it has no rule on file.

    Error ladder: 503 (database / llama-server down), 502 (embedding/output).
    """
    client = LlamaCppClient()

    try:
        result = await answer_question(req.question, client=client, seed=req.seed)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
    except (LlamaCppConnectionError, ModelResponseError) as exc:
        raise _llm_http_error(exc) from exc

    return AdvisorAskResponse(
        answer=result["answer"],
        model=result["model"],
        context_char_count=result["context_char_count"],
        sources=[
            AdvisorSource(
                id=m["id"],
                similarity=m["similarity"],
                metadata=m.get("metadata") or {},
                content=m["content"],
            )
            for m in result["matches"]
        ],
    )
