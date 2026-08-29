"""Health/status probes. Mounted *unversioned* (no ``/v1``) so infra checks — Docker
healthchecks, uptime monitors, `curl http://localhost:8000/health` — never break when the
API version advances."""

from fastapi import APIRouter

from app.services.vllm_client import VllmClient

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/llm")
async def llm_health():
    """Verify llama-server is reachable and has the configured model loaded, without
    running inference. Hits ``/health`` and ``/v1/models`` only, never ``/v1/chat/completions``.
    """
    client = VllmClient()
    ok, detail = await client.health()
    return {"ok": ok, "detail": detail, "model": client.model}
