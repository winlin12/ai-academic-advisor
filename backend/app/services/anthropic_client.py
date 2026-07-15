"""Anthropic API client for the advisor's chat side (replaces the Ollama transport).

Thin by design: model and output cap come from config, the API key comes from the process
environment (``ANTHROPIC_API_KEY``) via the SDK's own resolution — it is deliberately not a
Settings field so it can never be dumped into a log line. Embeddings are NOT here (the
Anthropic API has no embeddings endpoint); see ``services/rag/embeddings.py``.

Two generation surfaces:

    generate()  -- free-text answer (ask / explain-plan).
    propose()   -- schema-enforced structured output (revise-plan). The API guarantees the
                   response validates against the Pydantic model, which is what deleted the
                   old JSON-repair path in advisor_agent.

Transport errors (auth, rate limit, 5xx) are the SDK's typed ``anthropic.APIError`` family
and propagate to the routers, which map them onto the HTTP error ladder. ``max_retries``
(429/5xx backoff) is built into the SDK; we don't reimplement it.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelResponseError(RuntimeError):
    """The API answered, but not with usable content (refusal or truncated output)."""


class MissingAPIKeyError(RuntimeError):
    """No Anthropic credentials resolved. The SDK reports this as a bare TypeError at
    request time (construction succeeds with ``api_key=None``); we normalize it so the
    routers can map it to a clean 503 instead of a 500."""


_MISSING_KEY_DETAIL = (
    "No Anthropic API credentials found — set ANTHROPIC_API_KEY in backend/.env "
    "(or the process environment) and restart the backend."
)


class AnthropicClient:
    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        self.model = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.anthropic_max_tokens
        self._client = anthropic.AsyncAnthropic()

    @staticmethod
    def system_block(system_prompt: str) -> list[dict]:
        """The static system prompt as a cacheable prefix block.

        ``cache_control`` marks everything up to here for prompt caching (reads bill at
        ~0.1x input price). The minimum cacheable prefix is model-dependent (1,024 tokens
        on claude-sonnet-4-5; 4,096 on Haiku 4.5 and Opus); below the minimum the marker
        is silently ignored at no cost, so it is safe to leave on unconditionally. Only
        the /ask flow's system prompt (instructions + inlined DB schema) clears the
        minimum today — explain-plan and revise-plan prompts are ~100 tokens and don't.
        Nothing volatile (dates, student names, retrieved context) may go in the system
        prompt — that all belongs in the user turn, after the cache breakpoint.
        """
        return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    def _log_usage(self, endpoint: str, response: anthropic.types.Message) -> None:
        # The cost dashboard: an input-token spike here flags a context-builder bug long
        # before the invoice does.
        usage = response.usage
        logger.info(
            "anthropic %s: input=%d output=%d cache_write=%d cache_read=%d model=%s",
            endpoint,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens or 0,
            usage.cache_read_input_tokens or 0,
            response.model,
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Free-text generation grounded by the caller-supplied prompts."""
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_block(system_prompt),
                messages=[{"role": "user", "content": user_prompt}],
            )
        except TypeError as exc:
            # Our kwargs are static and correct, so a TypeError here is the SDK failing
            # to resolve credentials — not a programming error at this call site.
            raise MissingAPIKeyError(_MISSING_KEY_DETAIL) from exc
        self._log_usage("generate", response)
        if response.stop_reason == "refusal":
            raise ModelResponseError("The model declined to answer this request.")
        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip()

    async def propose(self, system_prompt: str, user_prompt: str, output_type: type[T]) -> T:
        """Schema-enforced generation: the response is guaranteed to validate as ``output_type``.

        Raises ``ModelResponseError`` when no parseable output came back (refusal, or the
        response was truncated by ``max_tokens``) so callers can fall back gracefully.
        """
        try:
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_block(system_prompt),
                messages=[{"role": "user", "content": user_prompt}],
                output_format=output_type,
            )
        except TypeError as exc:
            raise MissingAPIKeyError(_MISSING_KEY_DETAIL) from exc
        self._log_usage("propose", response)
        if response.parsed_output is None:
            raise ModelResponseError(
                f"No structured output returned (stop_reason={response.stop_reason})."
            )
        return response.parsed_output

    async def health(self) -> tuple[bool, str]:
        """Key-validity + reachability probe that consumes zero tokens (Models API)."""
        try:
            await self._client.models.retrieve(self.model)
        except TypeError:
            return False, _MISSING_KEY_DETAIL
        except anthropic.AuthenticationError:
            return False, (
                "Anthropic API authentication failed — is ANTHROPIC_API_KEY set in "
                "backend/.env (or the process environment)?"
            )
        except anthropic.NotFoundError:
            return False, f"Model id {self.model!r} was not found — check ANTHROPIC_MODEL."
        except Exception as exc:  # connection errors, 5xx, ...
            return False, str(exc)
        return True, f"Anthropic API reachable; model {self.model!r} available."
