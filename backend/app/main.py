import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_v1_router, system_router
from app.services.rag import store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Best-effort: create the pgvector RAG table so a fresh DB works without a manual step.
    # Deliberately swallow failures — the DB may legitimately be down at boot, and the app's
    # non-RAG routes must still start. The RAG routes degrade gracefully (empty retrieval)
    # until the table exists, so nothing here is allowed to block startup.
    try:
        store.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping RAG schema init at startup (DB unavailable?): %s", exc)
    yield


app = FastAPI(
    title="BoilerAdvisor",
    version="0.3.0",
    description="Local-first Purdue academic planning assistant backend.",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "system", "description": "Health and runtime probes (unversioned)."},
        {"name": "academic", "description": "Read-only catalog data: facets, programs, course search."},
        {"name": "planning", "description": "Deterministic planner: generate and directly edit plans. No LLM."},
        {"name": "advisor", "description": "LLM routes (Anthropic API): RAG ask, explain-plan, revise-plan."},
        {"name": "students", "description": "Student profile and saved-plan persistence."},
        {"name": "admin", "description": "Read-only database browsing for the admin page."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health probes stay at the root so infra checks never chase API versions; all product
# routes live under /v1 (see app/api/routes.py).
app.include_router(system_router)
app.include_router(api_v1_router)


@app.get("/", tags=["system"])
def root():
    return {
        "name": "BoilerAdvisor",
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/v1",
    }