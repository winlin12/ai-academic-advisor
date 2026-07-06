import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
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
    title="AI Academic Advisor",
    version="0.1.0",
    description="Local-first academic planning assistant backend.",
    lifespan=lifespan,
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

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "AI Academic Advisor",
        "status": "running",
        "docs": "/docs",
    }