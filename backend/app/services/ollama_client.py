from __future__ import annotations

import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import settings


TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Reasoning models served by Ollama (e.g. lfm2.5, deepseek-r1) emit their chain of thought
# in a leading <think>...</think> block. Strip it so callers get just the final answer.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


class LocalModelEndpointError(ValueError):
    pass


class EmbeddingError(RuntimeError):
    """Raised when the embedding endpoint returns a malformed or wrong-sized vector."""


class ModelJSONError(RuntimeError):
    """Raised when a ``format=json`` generation does not return parseable JSON."""


def _is_local_hostname(hostname: str) -> bool:
    lowered = hostname.lower().strip()
    if lowered in {"localhost", "host.docker.internal", "host.containers.internal"}:
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

        return strip_reasoning(data.get("response", ""))

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Like ``generate`` but constrain the model to emit a single JSON object.

        Ollama's ``format:"json"`` grammar-constrains decoding to valid JSON, which is what
        makes an 8B local model usable as a structured-output engine (the advisor agent then
        validates the object against a Pydantic schema). We still ``strip_reasoning`` first in
        case a reasoning model slips a ``<think>`` block in front of the JSON, and raise
        ``ModelJSONError`` on anything unparseable so the caller can fall back gracefully.
        """
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
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw = strip_reasoning(data.get("response", ""))
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelJSONError(f"Model did not return valid JSON: {exc}: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise ModelJSONError(f"Model returned JSON but not an object: {type(parsed).__name__}")
        return parsed

    async def embed(self, text: str, *, task_type: str = "search_document") -> list[float]:
        """Turn one text chunk into an embedding vector via the local embedding model.

        ``nomic-embed-text`` is an *asymmetric* retrieval model: it expects a task prefix so
        that a stored passage and a question about that passage land close together in vector
        space. We prepend ``search_document:`` when embedding catalog text for storage and
        ``search_query:`` when embedding a student's question. Everything else — base URL,
        the local-only endpoint guard, the JSON transport — is shared with ``generate``; only
        the model (``rag_embed_model``) and endpoint (``/api/embeddings``) differ.
        """
        prefix = f"{task_type}: " if task_type else ""
        payload = {
            "model": settings.rag_embed_model,
            "prompt": f"{prefix}{text}",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()

        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError(
                f"Embedding model '{settings.rag_embed_model}' returned no vector. "
                f"Is it pulled? Try: ollama pull {settings.rag_embed_model}"
            )
        if len(vector) != settings.rag_embed_dimensions:
            raise EmbeddingError(
                f"Embedding model '{settings.rag_embed_model}' returned {len(vector)} dims, "
                f"but rag_embed_dimensions is {settings.rag_embed_dimensions}. Fix the config "
                f"and the VECTOR(n) column so all three agree."
            )
        return vector
