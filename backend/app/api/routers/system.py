"""Health/status probes. Mounted *unversioned* (no ``/v1``) so infra checks — Docker
healthchecks, uptime monitors, `curl http://localhost:8000/health` — never break when the
API version advances."""

from fastapi import APIRouter

from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/llm")
async def llm_health():
    """Verify Ollama is reachable and the configured model is actually pulled, without
    running inference. Hits ``/api/tags`` only — same zero-cost intent as the old
    Anthropic Models-API probe, just against the local server instead of the cloud.
    """
    client = OllamaClient()
    ok, detail = await client.health()
    return {"ok": ok, "detail": detail, "model": client.model}
