"""vLLM HTTP client for the advisor's chat side (replaces the llama.cpp transport, which
replaced Ollama before it).

Production target: the FastAPI process and the vLLM server run on the SAME box (now a single
RTX 3090 Ti 24GB, replacing the RTX 3060 + RTX 2060 Super pair) — so the default base URL is
loopback. This is not a "local for dev only" shortcut; it is the real production topology.

WHY THE ENGINE CHANGED, 2026-08-22. llama.cpp pre-allocated its context window and DIVIDED it
between `--parallel` slots, so serving a second student concurrently halved the window each of
them got — and measured ~18% slower in aggregate than serving them one at a time (see
model_eval/config.yaml's `parallel`). The deployed server therefore ran one slot, and a second
student simply queued. vLLM pages the KV cache by the block: every concurrent request gets the
full `--max-model-len`, and concurrency is bounded by the size of the KV pool instead. For an
app whose entire shape is "several students on a site at once", that is the difference that
matters. Prefix caching is the second win — every request here opens with the same system
prompt and the same rendered requirement export, which is now prefilled once rather than per
call.

Talks to the OpenAI-compatible ``/v1/chat/completions`` endpoint — the model's own chat
template is applied server-side from the checkpoint's tokenizer config, so callers only ever
hand over system/user text, never a hand-rolled prompt string. The surface is unchanged from
the llama.cpp client, so callers (``advisor_agent.py``, ``rag/pipeline.py``, the ``advisor``
router) needed no logic changes:

    generate()  -- free-text answer (ask / explain-plan).
    propose()   -- schema-enforced structured output (revise-plan). vLLM constrains decoding
                   to the schema through xgrammar rather than llama.cpp's GBNF, but the
                   contract is the same and so is the caveat: it does not enforce every schema
                   keyword (e.g. minLength/pattern), so ``propose()`` keeps the same
                   Pydantic-validate-then-retry-once mitigation.
    health()    -- reachability + "is the loaded model the configured one" probe, zero
                   inference (hits /health and /v1/models, never /v1/chat/completions).

The context size and the loaded model are still fixed when the server is *launched*
(``--max-model-len``, ``--model``) — there is no per-request context override and no dynamic
model pull/unload. If the configured VLLM_MODEL doesn't match what the running server was
started with, health() flags it; fixing it means relaunching the server (see
services/model_manager.py, which owns that), not changing a request payload.

TWO TIMING NOTES for anyone reading the logs. There is no ``timings`` block in a vLLM response
— llama.cpp's ``prompt_ms``/``predicted_ms`` are simply gone — so ``_log_usage`` reports wall
clock instead. And a slow request no longer means a slow model: it can now mean queued behind
other students, a state the single-slot llama.cpp deployment could not be in.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
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
    """VLLM_BASE_URL points somewhere that isn't local/self-hosted while
    vllm_local_only is set — a misconfiguration guard, not a security boundary."""


class VllmConnectionError(RuntimeError):
    """Could not reach the vLLM server at all (down, wrong port, network)."""


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


class VllmClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vllm_model
        self.temperature = temperature if temperature is not None else settings.vllm_temperature
        self.max_tokens = max_tokens or settings.vllm_max_tokens
        self.timeout = timeout if timeout is not None else settings.vllm_timeout_s

        if settings.vllm_local_only and not is_local_model_endpoint(self.base_url):
            raise LocalModelEndpointError(
                "VLLM_BASE_URL must point to a local or self-hosted model endpoint. "
                "Allowed examples include localhost, LAN/private IPs, Tailscale IPs, "
                "host.docker.internal, and local Docker service names. Set "
                "VLLM_LOCAL_ONLY=false to lift this guard."
            )

    def _log_usage(self, endpoint: str, data: dict[str, Any]) -> None:
        # The cost/capacity dashboard: an unexpected completion_tokens spike flags a
        # context-builder bug, and a rising elapsed on already-warm requests flags GPU
        # contention — which under vLLM now includes "queued behind other students", a state
        # the single-slot llama.cpp deployment could not be in.
        #
        #
        # NO prompt_ms/predicted_ms. Those were llama.cpp's own counters, returned in a
        # `timings` block vLLM does not send; reading them here logged None on every request
        # for a while before this was noticed. Wall-clock elapsed is what is left, and it is
        # the number that matches what the student experienced anyway.
        usage = data.get("usage") or {}
        logger.info(
            "vllm %s: model=%s prompt_tokens=%s completion_tokens=%s elapsed_s=%.2f",
            endpoint,
            self.model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            data.get("_elapsed_s", float("nan")),
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
        # they dislike would get the same plan back. vLLM takes a per-request seed;
        # callers vary it (services/ai_planner.py bumps it per attempt) to resample properly
        # rather than by nudging temperature, which would also change answer quality.
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format

        timeout = self.timeout
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise VllmConnectionError(
                f"Cannot reach the vLLM server at {self.base_url} — is it running? "
                f"(production target: co-located on the same box as this backend)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise VllmConnectionError(
                f"vLLM server at {self.base_url} timed out after {timeout:g}s "
                f"(model={self.model})"
            ) from exc

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelResponseError(
                f"vLLM server at {self.base_url} returned {resp.status_code}: "
                f"{resp.text[:300]}"
            ) from exc
        data = resp.json()
        # Wall clock for this request, carried on the response so `_log_usage` can report it
        # without every call site threading a timer through. Underscore-prefixed because it is
        # NOT part of the OpenAI response shape and nothing but the logger should read it.
        data["_elapsed_s"] = time.perf_counter() - started
        return data

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

        ``response_format={"type": "json_schema", ...}`` compiles the schema into a decoding
        constraint server-side (xgrammar under vLLM, GBNF under the llama.cpp client this
        replaces) — but still no hard guarantee on every schema keyword. One retry with the
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
                f"Cannot reach the vLLM server at {self.base_url}. Is it running? "
                f"(`vllm serve <checkpoint> --served-model-name {self.model} ...`, or check "
                f"the systemd service.)"
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
                f"vLLM reachable at {self.base_url}; could not confirm the loaded model "
                f"({exc})."
            )

        # AN EXACT MATCH ON THE SERVED NAME, not a basename comparison. Under llama.cpp
        # /v1/models reported the gguf PATH the process was launched with, so the check had to
        # reduce both sides to a filename. vLLM reports `--served-model-name` verbatim — the
        # same short name config.py holds — so any difference here is a real mismatch rather
        # than a path-shape artifact.
        loaded_ids = [m.get("id", "") for m in data.get("data", [])]
        if self.model not in loaded_ids:
            return False, (
                f"vLLM is reachable at {self.base_url} but the loaded model "
                f"({loaded_ids or 'unknown'}) doesn't match the configured {self.model!r}. "
                f"Relaunch it with `--served-model-name {self.model}`."
            )
        return True, f"vLLM reachable at {self.base_url}; model {self.model!r} is loaded."
