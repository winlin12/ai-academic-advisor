import psycopg
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AcademicFacetResponse,
    AcademicProgramDetail,
    AcademicProgramSummary,
    AdvisorAskRequest,
    AdvisorAskResponse,
    AdvisorSource,
    ExplainPlanRequest,
    PlanResponse,
    StudentProfile,
)
from app.services.academic_db import (
    fetch_academic_facets,
    fetch_program_detail,
    fetch_program_summaries,
    search_courses,
)
from app.services.catalog import load_catalog
from app.services.ollama_client import EmbeddingError, LocalModelEndpointError, OllamaClient
from app.services.planner import generate_plan
from app.services.rag.pipeline import answer_question

router = APIRouter()


def academic_db_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Academic database unavailable: {exc}")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ollama")
async def ollama_health():
    try:
        client = OllamaClient()
    except LocalModelEndpointError as exc:
        return {
            "ok": False,
            "detail": str(exc),
            "local_only": True,
            "compute_warning": "Local models can use substantial CPU/GPU, memory, and battery.",
        }

    ok, detail = await client.health()
    return {
        "ok": ok,
        "detail": detail,
        "ollama_url": client.base_url,
        "model": client.model,
        "local_only": client.local_only,
        "compute_warning": "Local models can use substantial CPU/GPU, memory, and battery.",
    }


@router.get("/catalog/courses")
def list_courses():
    return {"courses": [course.model_dump() for course in load_catalog()]}


@router.get("/academic/facets", response_model=AcademicFacetResponse)
def academic_facets():
    try:
        return fetch_academic_facets()
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc


@router.get("/academic/programs", response_model=list[AcademicProgramSummary])
def academic_programs(
    query: str | None = None,
    catalog_year: int | None = None,
    school: str | None = None,
    limit: int = 120,
):
    try:
        return fetch_program_summaries(
            query=query,
            catalog_year=catalog_year,
            school=school,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc


@router.get("/academic/programs/{program_id}", response_model=AcademicProgramDetail)
def academic_program_detail(program_id: str):
    try:
        program = fetch_program_detail(program_id)
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc

    if program is None:
        raise HTTPException(status_code=404, detail="Academic program not found")
    return program


@router.get("/academic/courses/search")
def academic_course_search(
    query: str | None = None,
    subject: str | None = None,
    limit: int = 80,
):
    try:
        courses = search_courses(query=query, subject=subject, limit=limit)
        return {"courses": [course.model_dump() for course in courses]}
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc


@router.post("/plan/generate", response_model=PlanResponse)
def plan_generate(profile: StudentProfile):
    catalog = load_catalog()
    return generate_plan(profile, catalog)


@router.post("/advisor/explain-plan")
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
{req.plan}

Explain the plan clearly. Focus on prerequisites, semester sequencing, warnings, and risk.
""".strip()

    try:
        answer = await client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    return {"answer": answer}


@router.post("/advisor/ask", response_model=AdvisorAskResponse)
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
