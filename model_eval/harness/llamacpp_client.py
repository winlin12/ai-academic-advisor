"""Minimal llama-server HTTP client (stdlib only — the harness must not depend on the app).

Replaces the old Ollama client. Three things changed and all of them matter:

1. ENDPOINT. The app talks to llama-server's OpenAI-compatible ``/v1/chat/completions`` and
   lets the gguf's own chat template (applied server-side under ``--jinja``) format the turn.
   The harness does the same. It deliberately does NOT use llama.cpp's native ``/completion``,
   even though that endpoint's timing fields are nicer — a hand-rolled prompt string would
   measure a prompt format the app never sends.

2. TIMINGS. Ollama returned nanosecond counters in its final stream chunk. llama-server
   returns an OpenAI-shaped stream, so we ask for ``stream_options.include_usage`` (token
   counts) and ``timings_per_token`` (llama.cpp's own prompt_ms/predicted_ms) and fall back
   to wall-clock when a build omits them. TTFT is always wall-clock off the stream.

3. CONTEXT SIZE IS NOT A REQUEST PARAMETER. Ollama took ``num_ctx`` per request; llama-server
   fixes it at launch (``--ctx-size``). The harness therefore owns the server process
   (see ``server.py``) instead of assuming someone started it correctly — the alternative is
   silently comparing models at different context sizes, which is exactly the pollution this
   harness exists to prevent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class LlamaCppError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    ttft_s: float | None            # wall-clock, request sent -> first content token
    total_s: float                  # wall-clock, request sent -> stream closed
    eval_count: int | None          # completion tokens
    prompt_eval_count: int | None   # prompt tokens
    prompt_ms: float | None         # llama.cpp's own prompt-eval time
    predicted_ms: float | None      # llama.cpp's own generation time
    finish_reason: str | None = None
    raw_final: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling instead of stopping. On the plan task this is a real and
        common failure mode (a whole 8-semester schedule is a lot of JSON), so it is recorded
        rather than left to look like a malformed response."""
        return self.finish_reason == "length"


class LlamaCppClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None):
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            raise LlamaCppError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LlamaCppError(
                f"Cannot reach llama-server at {self.base_url} ({exc.reason})."
            ) from exc

    # --- probes ---------------------------------------------------------------------------

    def health(self, timeout: float = 5.0) -> bool:
        try:
            with self._request("GET", "/health", timeout=timeout) as resp:
                return json.load(resp).get("status") == "ok"
        except (LlamaCppError, json.JSONDecodeError):
            return False

    def props(self) -> dict[str, Any]:
        """Server-side truth about what is actually loaded: context size, model path, build.

        This is the check Ollama never made possible — with llama.cpp the context size is a
        launch flag, so "is every model really running at num_ctx?" is answerable instead of
        assumed. ``runner`` records it per model and the report flags mismatches.
        """
        try:
            with self._request("GET", "/props", timeout=10) as resp:
                return json.load(resp)
        except (LlamaCppError, json.JSONDecodeError):
            return {}

    def loaded_model(self) -> str | None:
        try:
            with self._request("GET", "/v1/models", timeout=10) as resp:
                data = json.load(resp)
        except (LlamaCppError, json.JSONDecodeError):
            return None
        entries = data.get("data") or []
        return entries[0].get("id") if entries else None

    # --- generation -----------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        options: dict[str, Any],
        think: bool | None = None,
        response_format: dict | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """One streamed chat completion. Same message shape the app sends."""
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "timings_per_token": True,
            **options,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        # Thinking is disabled at LAUNCH via `--reasoning off` (see server.py) because that is
        # the only switch that works uniformly across templates. This per-request kwarg is the
        # belt to that suspenders: Qwen3-family templates read `enable_thinking`, gpt-oss reads
        # `reasoning_effort`. Anything the model still emits is stripped by the scorers.
        kwargs = dict(chat_template_kwargs or {})
        if think is False:
            kwargs.setdefault("enable_thinking", False)
        if kwargs:
            payload["chat_template_kwargs"] = kwargs

        start = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        finish_reason: str | None = None

        with self._request("POST", "/v1/chat/completions", payload) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LlamaCppError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        return GenerationResult(
            text="".join(parts).strip(),
            ttft_s=ttft,
            total_s=time.perf_counter() - start,
            eval_count=usage.get("completion_tokens") or timings.get("predicted_n"),
            prompt_eval_count=usage.get("prompt_tokens") or timings.get("prompt_n"),
            prompt_ms=timings.get("prompt_ms"),
            predicted_ms=timings.get("predicted_ms"),
            finish_reason=finish_reason,
            raw_final={"usage": usage, "timings": timings},
        )
