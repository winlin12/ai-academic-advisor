from fastapi import APIRouter, HTTPException

from app.models.schemas import ExplainPlanRequest, PlanResponse, StudentProfile
from app.services.catalog import load_catalog
from app.services.ollama_client import OllamaClient
from app.services.planner import generate_plan

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ollama")
async def ollama_health():
    client = OllamaClient()
    ok, detail = await client.health()
    return {"ok": ok, "detail": detail}


@router.get("/catalog/courses")
def list_courses():
    return {"courses": [course.model_dump() for course in load_catalog()]}


@router.post("/plan/generate", response_model=PlanResponse)
def plan_generate(profile: StudentProfile):
    catalog = load_catalog()
    return generate_plan(profile, catalog)


@router.post("/advisor/explain-plan")
async def explain_plan(req: ExplainPlanRequest):
    client = OllamaClient()

    system_prompt = """
You are an AI academic planning assistant.
You are not an official academic advisor.
You must not invent courses, prerequisites, requirements, or policies.
Explain only from the supplied structured plan.
Always recommend verifying important decisions with an official advisor.
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
