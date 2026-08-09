import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_v1_router, system_router
from app.core.config import settings
from app.services.model_manager import ModelManager
from app.services.rag import store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort: create the pgvector RAG table so a fresh DB works without a manual step.
    # Deliberately swallow failures — the DB may legitimately be down at boot, and the app's
    # non-RAG routes must still start. The RAG routes degrade gracefully (empty retrieval)
    # until the table exists, so nothing here is allowed to block startup.
    try:
        store.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping RAG schema init at startup (DB unavailable?): %s", exc)

    # Launches the default local model (see services/model_manager.py) — the backend now owns
    # llama-server's process lifecycle so a user can switch models from the UI.
    #
    # BACKGROUNDED, NOT AWAITED. A model launch can take anywhere from a few seconds to the
    # full `llamacpp_startup_timeout_s` (180s default) — GPU contention, a cold disk read of a
    # 15-20GB gguf, whatever. Awaiting it here would block uvicorn's own startup on it, which
    # means catalog browsing and the deterministic planner — routes that need no LLM at all —
    # would be unreachable for up to three minutes on every cold start. `GET /v1/models`
    # reports `switching_to`/`last_error` while this runs in the background, so the frontend
    # can show progress instead of the whole app looking down.
    app.state.model_manager = ModelManager(settings=settings)
    startup_task = asyncio.create_task(app.state.model_manager.start_default())
    yield
    startup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await startup_task
    await app.state.model_manager.stop()


app = FastAPI(
    title="BoilerAdvisor",
    version="0.3.0",
    description="Local-first Purdue academic planning assistant backend.",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "system", "description": "Health and runtime probes (unversioned)."},
        {"name": "academic", "description": "Read-only catalog data: facets, programs, course search."},
        {"name": "planning", "description": "Deterministic planner: generate and directly edit plans. No LLM."},
        {"name": "advisor", "description": "LLM routes (local llama.cpp): RAG ask, explain-plan, revise-plan."},
        {"name": "models", "description": "Which local model is running, and switching it."},
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