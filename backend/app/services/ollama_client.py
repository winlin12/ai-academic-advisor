from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx

from app.core.config import settings


TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class LocalModelEndpointError(ValueError):
    pass


def _is_local_hostname(hostname: str) -> bool:
    lowered = hostname.lower().strip()
    if lowered in {"localhost", "host.docker.internal"}:
        return True
    if lowered.endswith(".localhost") or lowered.endswith(".local"):
        return True
    return "." not in lowered


def _is_local_ip(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address in TAILSCALE_CGNAT
    )


def is_local_model_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname:
        return False
    return _is_local_ip(parsed.hostname) or _is_local_hostname(parsed.hostname)


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.local_only = settings.ollama_local_only

        if self.local_only and not is_local_model_endpoint(self.base_url):
            raise LocalModelEndpointError(
                "OLLAMA_BASE_URL must point to a local or self-hosted model endpoint. "
                "Allowed examples include localhost, LAN/private IPs, Tailscale IPs, "
                "host.docker.internal, and local Docker service names."
            )

    async def health(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return True, f"Connected to local Ollama-compatible endpoint at {self.base_url}"
        except Exception as exc:
            return False, str(exc)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"""
SYSTEM:
{system_prompt}

USER:
{user_prompt}
""".strip()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("response", "")
