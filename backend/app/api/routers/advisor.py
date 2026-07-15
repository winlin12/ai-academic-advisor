"""LLM-facing routes: ask (RAG), explain-plan, revise-plan. These are the only routes that
spend Anthropic API tokens; everything they return that *matters* (the plan itself) is still
produced or re-validated by the deterministic planner."""

import anthropic
import psycopg
from fastapi import APIRouter, HTTPException

from app.api.deps import academic_db_unavailable, resolve_for_planning
from app.models.schemas import (
    AdvisorAskRequest,
    AdvisorAskResponse,
    AdvisorSource,
    ExplainPlanRequest,
    ExplainPlanResponse,
    RevisePlanRequest,
    RevisePlanResponse,
)
from app.services.advisor_agent import revise_plan
from app.services.anthropic_client import AnthropicClient, MissingAPIKeyError, ModelResponseError
from app.services.rag.embeddings import EmbeddingError
from app.services.rag.pipeline import answer_question

router = APIRouter(prefix="/advisor", tags=["advisor"])


def _llm_http_error(exc: anthropic.APIError | MissingAPIKeyError) -> HTTPException:
    """Map the SDK's typed errors (plus our own MissingAPIKeyError) onto the HTTP ladder,
    most specific first.

    503 = our config problem (key), 429 = upstream rate limit (retryable by the client),
    502 = upstream transport/API failure. The SDK already retried 429/5xx with backoff
    before any of these surface here.
    """
    if isinstance(exc, MissingAPIKeyError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, anthropic.AuthenticationError):
        return HTTPException(
            status_code=503,
            detail=(
                "Anthropic API authentication failed — set ANTHROPIC_API_KEY in "
                "backend/.env (or the process environment) and restart the backend."
            ),
        )
    if isinstance(exc, anthropic.RateLimitError):
        return HTTPException(status_code=429, detail="Anthropic API rate limit hit; retry shortly.")
    if isinstance(exc, anthropic.APIStatusError):
        return HTTPException(status_code=502, detail=f"Anthropic API error {exc.status_code}: {exc.message}")
    return HTTPException(status_code=502, detail=f"Could not reach the Anthropic API: {exc}")


@router.post("/explain-plan", response_model=ExplainPlanResponse)
async def explain_plan(req: ExplainPlanRequest):
    client = AnthropicClient()

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
        answer = await client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    except (anthropic.APIError, MissingAPIKeyError) as exc:
        raise _llm_http_error(exc) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExplainPlanResponse(answer=answer)


@router.post("/revise-plan", response_model=RevisePlanResponse)
async def revise_plan_route(req: RevisePlanRequest):
    """Revise a plan from free-text feedback via the structured-output agent.

    The model proposes edits (reorder/defer/avoid-tags/credit-cap) as a schema-enforced
    PlanEditProposal and the deterministic planner re-validates them, so the returned plan
    is always legal. Error ladder: 503 (API key), 429 (upstream rate limit), 502 (API).
    """
    client = AnthropicClient()

    profile, catalog = resolve_for_planning(req.profile)
    try:
        return await revise_plan(profile, catalog, req.feedback, client=client)
    except (anthropic.APIError, MissingAPIKeyError) as exc:
        raise _llm_http_error(exc) from exc


@router.post("/ask", response_model=AdvisorAskResponse)
async def advisor_ask(req: AdvisorAskRequest):
    """Answer a free-text student question via exact + semantic (pgvector) retrieval.

    Course codes named in the question are fetched deterministically; the question is also
    embedded (in-process) and the most similar ``academic_rules`` chunks are pulled by
    cosine distance. Claude is grounded on just those, under a hard context token budget.
    The retrieved chunks come back as ``sources`` so the answer stays inspectable and
    citable. The catalog must first be loaded into ``academic_rules``
    (``python -m app.services.rag.ingest_catalog``), otherwise retrieval returns nothing
    and the model will say it has no rule on file.

    Error ladder: 503 (API key / database), 429 (upstream rate limit), 502 (embedding/API).
    """
    client = AnthropicClient()

    try:
        result = await answer_question(req.question, client=client)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
    except (anthropic.APIError, MissingAPIKeyError) as exc:
        raise _llm_http_error(exc) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

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
