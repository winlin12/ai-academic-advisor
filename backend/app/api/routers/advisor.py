"""LLM-facing routes: ask (RAG), explain-plan, revise-plan. These are the only routes that
spend local model compute; everything they return that *matters* (the plan itself) is still
produced or re-validated by the deterministic planner."""

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
from app.services.ollama_client import EmbeddingError, LocalModelEndpointError, OllamaClient
from app.services.rag.pipeline import answer_question

router = APIRouter(prefix="/advisor", tags=["advisor"])


@router.post("/explain-plan", response_model=ExplainPlanResponse)
async def explain_plan(req: ExplainPlanRequest):
    try:
        client = OllamaClient()
    except LocalModelEndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    system_prompt = """
You are an AI academic planning assistant.
You are not an official academic advisor.
You must not invent courses, prerequisites, requirements, or policies.
Explain only from the supplied structured plan.
Always recommend verifying important decisions with an official advisor.
You are running on a local or student-owned model endpoint, so keep answers concise
unless the student asks for depth.
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
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    return ExplainPlanResponse(answer=answer)


@router.post("/revise-plan", response_model=RevisePlanResponse)
async def revise_plan_route(req: RevisePlanRequest):
    """Revise a plan from free-text feedback via the local LFM2 agent.

    The model proposes edits (reorder/defer/avoid-tags/credit-cap) and the deterministic
    planner re-validates them, so the returned plan is always legal. Error ladder mirrors the
    other advisor routes: 400 (non-local endpoint), 502 (model transport / bad JSON).
    """
    try:
        client = OllamaClient()
    except LocalModelEndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile, catalog = resolve_for_planning(req.profile)
    try:
        return await revise_plan(profile, catalog, req.feedback, client=client)
    except Exception as exc:  # httpx / model transport failure
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc


@router.post("/ask", response_model=AdvisorAskResponse)
async def advisor_ask(req: AdvisorAskRequest):
    """Answer a free-text student question via semantic (pgvector) retrieval.

    The question is embedded, the top-k most similar ``academic_rules`` chunks are pulled by
    cosine distance, and the local model is grounded on just those. The retrieved chunks come
    back as ``sources`` so the answer stays inspectable and citable. The catalog must first be
    loaded into ``academic_rules`` (``python -m app.services.rag.ingest_catalog``), otherwise
    retrieval returns nothing and the model will say it has no rule on file.

    Error ladder: 400 (non-local endpoint), 503 (database), 502 (embedding/model transport).
    """
    try:
        client = OllamaClient()
    except LocalModelEndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await answer_question(req.question, client=client)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
    except Exception as exc:  # httpx / model transport failure
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

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
