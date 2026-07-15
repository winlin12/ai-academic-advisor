"""Ollama HTTP client for the advisor's chat side (replaces the Anthropic transport).

Production target: the FastAPI process and Ollama run on the SAME box (the RTX 2060
Super server, 24/7) — so the default base URL is plain ``localhost``. This is not a
"local for dev only" shortcut; it is the real production topology.

Same surface as the Anthropic client it replaces, so callers (``advisor_agent.py``,
``rag/pipeline.py``, the ``advisor`` router) barely change:

    generate()  -- free-text answer (ask / explain-plan).
    propose()   -- schema-enforced structured output (revise-plan). Unlike Anthropic's
                   ``messages.parse``, Ollama does not GUARANTEE the JSON matches the
                   schema even with ``format=<json schema>`` — it constrains decoding to
                   valid JSON syntax, not to the schema's required fields/types. So
                   ``propose()`` validates with Pydantic itself and does ONE retry with
                   the validation error appended before giving up (mirrors the
                   retry-on-invalid-SQL mitigation validated in ``model_eval/``).
    health()    -- reachability + "is this model actually pulled" probe, zero inference.

Model default is ``llama3.1:8b`` as a placeholder pending ``model_eval/``'s verdict on
which 8GB-class model to standardize on — swap ``OLLAMA_MODEL`` in ``backend/.env``
once that harness names a winner.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Reasoning models served by Ollama (deepseek-r1, Gemma 4's thinking mode) emit chain-of-
# thought in a leading <think>...</think> block. Strip it so callers get just the answer.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


class LocalModelEndpointError(ValueError):
    """OLLAMA_BASE_URL points somewhere that isn't local/self-hosted while
    ollama_local_only is set — a misconfiguration guard, not a security boundary."""


class OllamaConnectionError(RuntimeError):
    """Could not reach the Ollama server at all (down, wrong port, network)."""


class ModelNotFoundError(RuntimeError):
    """The configured model tag isn't pulled on this Ollama instance."""


class ModelResponseError(RuntimeError):
    """The server answered, but not with usable content (empty output, or — for
    propose() — JSON that still won't validate after one retry)."""


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


def _extract_json_object(text: str) -> str:
    """Ollama's format=<schema> keeps replies to JSON syntax but not to zero
    surrounding prose on every model; strip markdown fences some models still add."""
    cleaned = strip_reasoning(text)
    fenced = _FENCE_RE.search(cleaned)
    return fenced.group(1).strip() if fenced else cleaned


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        num_ctx: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.num_ctx = num_ctx or settings.ollama_num_ctx
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.keep_alive = keep_alive or settings.ollama_keep_alive

        if settings.ollama_local_only and not is_local_model_endpoint(self.base_url):
            raise LocalModelEndpointError(
                "OLLAMA_BASE_URL must point to a local or self-hosted model endpoint. "
                "Allowed examples include localhost, LAN/private IPs, Tailscale IPs, "
                "host.docker.internal, and local Docker service names. Set "
                "OLLAMA_LOCAL_ONLY=false to lift this guard."
            )

    def _options(self) -> dict[str, Any]:
        return {"num_ctx": self.num_ctx, "temperature": self.temperature}

    def _log_usage(self, endpoint: str, data: dict[str, Any]) -> None:
        # The cost/capacity dashboard: eval_count spikes or a rising load_duration on
        # already-warm requests flag a context-builder bug or GPU contention early.
        logger.info(
            "ollama %s: model=%s prompt_tokens=%s output_tokens=%s "
            "load_ms=%s eval_ms=%s",
            endpoint,
            self.model,
            data.get("prompt_eval_count"),
            data.get("eval_count"),
            (data.get("load_duration") or 0) // 1_000_000,
            (data.get("eval_duration") or 0) // 1_000_000,
        )

    async def _generate_raw(
        self, prompt: str, *, output_format: dict | str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(),
            "keep_alive": self.keep_alive,
        }
        if output_format is not None:
            payload["format"] = output_format

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url} — is it running? "
                f"(production target: co-located on the same box as this backend)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaConnectionError(
                f"Ollama at {self.base_url} timed out after 120s (model={self.model})"
            ) from exc

        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model {self.model!r} is not pulled on {self.base_url}. "
                f"Run: ollama pull {self.model}"
            )
        resp.raise_for_status()
        return resp.json()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Free-text generation grounded by the caller-supplied prompts."""
        prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        data = await self._generate_raw(prompt)
        self._log_usage("generate", data)
        text = strip_reasoning(data.get("response", ""))
        if not text:
            raise ModelResponseError("Model returned an empty response.")
        return text

    async def propose(self, system_prompt: str, user_prompt: str, output_type: type[T]) -> T:
        """Schema-constrained generation, validated against ``output_type``.

        Ollama's ``format=<json schema>`` constrains token decoding to valid JSON syntax
        but not to the schema's required fields/types/constraints — so unlike Anthropic's
        ``messages.parse``, validity is NOT guaranteed. One retry with the validation
        error appended (same mitigation validated in model_eval/) before raising.
        """
        schema = output_type.model_json_schema()
        prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

        for attempt in (1, 2):
            data = await self._generate_raw(prompt, output_format=schema)
            self._log_usage("propose", data)
            raw = _extract_json_object(data.get("response", ""))
            try:
                parsed = json.loads(raw)
                return output_type.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 2:
                    raise ModelResponseError(
                        f"No schema-valid output from {self.model} after retry: {exc}"
                    ) from exc
                logger.warning("propose(): invalid output on attempt 1, retrying: %s", exc)
                prompt = (
                    f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\n"
                    f"Your previous response did not match the required JSON schema.\n"
                    f"Previous response: {raw[:500]}\nError: {exc}\n"
                    f"Respond again with ONLY a JSON object matching the schema."
                )
        raise AssertionError("unreachable")  # loop always returns or raises above

    async def health(self) -> tuple[bool, str]:
        """Reachability + "is the configured model actually pulled" probe. Zero-token —
        hits /api/tags, never /api/generate."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            return False, (
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                f"(`ollama serve`, or check the systemd service on the server.)"
            )
        except Exception as exc:  # noqa: BLE001 - health probe must never raise
            return False, str(exc)

        tags = {m.get("name") for m in data.get("models", [])}
        # Ollama tags are name[:tag]; a bare model name also matches its ":latest".
        pulled = self.model in tags or f"{self.model}:latest" in tags
        if not pulled:
            return False, (
                f"Ollama is reachable at {self.base_url} but {self.model!r} is not "
                f"pulled. Run: ollama pull {self.model}"
            )
        return True, f"Ollama reachable at {self.base_url}; model {self.model!r} is pulled."
