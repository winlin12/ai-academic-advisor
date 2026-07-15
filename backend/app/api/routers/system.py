"""Health/status probes. Mounted *unversioned* (no ``/v1``) so infra checks — Docker
healthchecks, uptime monitors, `curl http://localhost:8000/health` — never break when the
API version advances."""

from fastapi import APIRouter

from app.services.anthropic_client import AnthropicClient

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/llm")
async def llm_health():
    """Verify the Anthropic API key and model id end-to-end without spending a token.

    Uses the Models API (``models.retrieve``) — a free endpoint — so this probe can run on
    every deploy and from uptime monitors at zero inference cost.
    """
    client = AnthropicClient()
    ok, detail = await client.health()
    return {"ok": ok, "detail": detail, "model": client.model}
