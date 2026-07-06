"""Health/status probes. Mounted *unversioned* (no ``/v1``) so infra checks — Docker
healthchecks, uptime monitors, `curl http://localhost:8000/health` — never break when the
API version advances."""

from fastapi import APIRouter

from app.services.ollama_client import LocalModelEndpointError, OllamaClient

router = APIRouter(tags=["system"])


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
