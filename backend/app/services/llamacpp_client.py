"""llama.cpp HTTP client for the advisor's chat side (replaces the Ollama transport).

Production target: the FastAPI process and llama.cpp's server (``llama-server``) run on the
SAME box (the RTX 2060 Super server, 24/7) — so the default base URL is plain ``localhost``.
This is not a "local for dev only" shortcut; it is the real production topology.

Talks to llama-server's OpenAI-compatible ``/v1/chat/completions`` endpoint — the model's own
chat template (Qwen2.5's ChatML) is applied server-side from the gguf's embedded metadata, so
callers only ever hand over system/user text, never a hand-rolled prompt string. Same surface
as the Ollama client it replaces, so callers (``advisor_agent.py``, ``rag/pipeline.py``, the
``advisor`` router) barely change:

    generate()  -- free-text answer (ask / explain-plan).
    propose()   -- schema-enforced structured output (revise-plan). llama.cpp's grammar-
                   constrained decoding (``response_format={"type": "json_schema", ...}``)
                   converts the schema to a GBNF grammar server-side — stricter than Ollama's
                   format=<schema>, which only constrained JSON *syntax* — but still doesn't
                   enforce every schema keyword (e.g. minLength/pattern), so ``propose()``
                   keeps the same Pydantic-validate-then-retry-once mitigation.
    health()    -- reachability + "is the loaded model the configured one" probe, zero
                   inference (hits /health and /v1/models, never /v1/chat/completions).

Unlike Ollama, llama.cpp's context size and the loaded model are fixed when ``llama-server``
is *launched* (``--ctx-size``, ``--model``) — there is no per-request num_ctx override and no
dynamic model pull/unload. If the configured LLAMACPP_MODEL doesn't match what the running
server was started with, health() flags it; fixing it means restarting llama-server, not
changing a request payload.

Model default is ``Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`` — model_eval/'s verdict for the
2060 Super's 8GB budget: best quality among the 8GB-class models tried, and code-tuned
pretraining measurably helped this app's SQL-shaped retrieval/summarization tasks.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Defensive only: grammar-constrained decoding means the model shouldn't emit chain-of-thought
# or markdown fences around JSON, but a future model swap (e.g. to a reasoning-tuned model)
# could reintroduce either, so keep both cleanups cheap and unconditional.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


class LocalModelEndpointError(ValueError):
    """LLAMACPP_BASE_URL points somewhere that isn't local/self-hosted while
    llamacpp_local_only is set — a misconfiguration guard, not a security boundary."""


class LlamaCppConnectionError(RuntimeError):
    """Could not reach llama-server at all (down, wrong port, network)."""


class ModelResponseError(RuntimeError):
    """The server answered, but not with usable content (empty output, a non-2xx status, or —
    for propose() — JSON that still won't validate after one retry).

    ``raw`` carries the model's last output when there was one. Callers that can do something
    with a partial answer use it: a plan of study truncated at the token ceiling still holds
    most of a valid schedule, and throwing the text away turns a recoverable draft into a total
    failure (see ai_planner._salvage_semesters).
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


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
    """Grammar-constrained output should already be a bare JSON object; strip markdown fences
    defensively in case a future model still wraps them despite the schema constraint."""
    cleaned = strip_reasoning(text)
    fenced = _FENCE_RE.search(cleaned)
    return fenced.group(1).strip() if fenced else cleaned


class LlamaCppClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or settings.llamacpp_base_url).rstrip("/")
        self.model = model or settings.llamacpp_model
        self.temperature = temperature if temperature is not None else settings.llamacpp_temperature
        self.max_tokens = max_tokens or settings.llamacpp_max_tokens
        self.timeout = timeout if timeout is not None else settings.llamacpp_timeout_s

        if settings.llamacpp_local_only and not is_local_model_endpoint(self.base_url):
            raise LocalModelEndpointError(
                "LLAMACPP_BASE_URL must point to a local or self-hosted model endpoint. "
                "Allowed examples include localhost, LAN/private IPs, Tailscale IPs, "
                "host.docker.internal, and local Docker service names. Set "
                "LLAMACPP_LOCAL_ONLY=false to lift this guard."
            )

    def _log_usage(self, endpoint: str, data: dict[str, Any]) -> None:
        # The cost/capacity dashboard: a rising prompt_ms on already-warm requests or an
        # unexpected completion_tokens spike flags a context-builder bug or GPU contention.
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        logger.info(
            "llamacpp %s: model=%s prompt_tokens=%s completion_tokens=%s "
            "prompt_ms=%s predicted_ms=%s",
            endpoint,
            self.model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            timings.get("prompt_ms"),
            timings.get("predicted_ms"),
        )

    async def _chat_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        # SEED IS WHAT MAKES "REGENERATE" MEAN ANYTHING. At temperature 0.15 an identical
        # request returns a near-identical answer, so a student pressing regenerate on a plan
        # they dislike would get the same plan back. llama-server takes a per-request seed;
        # callers vary it (services/ai_planner.py bumps it per attempt) to resample properly
        # rather than by nudging temperature, which would also change answer quality.
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format

        timeout = self.timeout
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise LlamaCppConnectionError(
                f"Cannot reach llama.cpp server at {self.base_url} — is llama-server running? "
                f"(production target: co-located on the same box as this backend)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LlamaCppConnectionError(
                f"llama.cpp server at {self.base_url} timed out after {timeout:g}s "
                f"(model={self.model})"
            ) from exc

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelResponseError(
                f"llama.cpp server at {self.base_url} returned {resp.status_code}: "
                f"{resp.text[:300]}"
            ) from exc
        return resp.json()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Free-text generation grounded by the caller-supplied prompts."""
        data = await self._chat_raw(system_prompt, user_prompt, seed=seed, max_tokens=max_tokens)
        self._log_usage("generate", data)
        content = data["choices"][0]["message"]["content"] or ""
        text = strip_reasoning(content)
        if not text:
            raise ModelResponseError("Model returned an empty response.")
        return text

    async def propose(
        self,
        system_prompt: str,
        user_prompt: str,
        output_type: type[T],
        schema: dict | None = None,
        *,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Schema-constrained generation, validated against ``output_type``.

        llama.cpp's ``response_format={"type": "json_schema", ...}`` converts the schema to a
        GBNF grammar and constrains decoding to it — stricter than Ollama's old format=<schema>
        (syntax-only), but still no hard guarantee on every schema keyword. One retry with the
        validation error appended (same mitigation validated in model_eval/) before raising.

        ``schema`` overrides the one derived from ``output_type``, for callers that can narrow
        it against the request — e.g. restricting course-code fields to an enum of the courses
        this particular student actually has left. The result is still validated against
        ``output_type``, so a narrowed schema can only ever be stricter than the model.
        """
        schema = schema or output_type.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": output_type.__name__, "schema": schema},
        }
        prompt = user_prompt

        for attempt in (1, 2):
            data = await self._chat_raw(
                system_prompt, prompt, response_format=response_format,
                # The retry must not resample the same tokens: with an identical seed and a
                # temperature this low, "try again" reproduces the malformed output verbatim
                # and burns a second generation to fail the same way.
                seed=None if seed is None else seed + attempt - 1,
                max_tokens=max_tokens,
            )
            self._log_usage("propose", data)
            content = data["choices"][0]["message"]["content"] or ""
            raw = _extract_json_object(content)
            try:
                parsed = json.loads(raw)
                return output_type.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 2:
                    raise ModelResponseError(
                        f"No schema-valid output from {self.model} after retry: {exc}",
                        raw=raw,
                    ) from exc
                logger.warning("propose(): invalid output on attempt 1, retrying: %s", exc)
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Your previous response did not match the required JSON schema.\n"
                    f"Previous response: {raw[:500]}\nError: {exc}\n"
                    f"Respond again with ONLY a JSON object matching the schema."
                )
        raise AssertionError("unreachable")  # loop always returns or raises above

    async def health(self) -> tuple[bool, str]:
        """Reachability + "is the loaded model the configured one" probe. Zero-token — hits
        /health and /v1/models, never /v1/chat/completions."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
        except httpx.ConnectError:
            return False, (
                f"Cannot reach llama.cpp server at {self.base_url}. Is it running? "
                f"(`llama-server -m {self.model} ...`, or check the systemd service.)"
            )
        except Exception as exc:  # noqa: BLE001 - health probe must never raise
            return False, str(exc)

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 - /health passed; this is best-effort extra info
            return True, (
                f"llama.cpp reachable at {self.base_url}; could not confirm the loaded model "
                f"({exc})."
            )

        loaded_ids = [m.get("id", "") for m in data.get("data", [])]
        wanted = Path(self.model).name
        if not any(Path(loaded_id).name == wanted for loaded_id in loaded_ids):
            return False, (
                f"llama.cpp is reachable at {self.base_url} but the loaded model "
                f"({loaded_ids or 'unknown'}) doesn't match the configured {self.model!r}. "
                f"Restart llama-server with `-m {self.model}`."
            )
        return True, f"llama.cpp reachable at {self.base_url}; model {self.model!r} is loaded."
