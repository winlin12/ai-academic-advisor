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
    # llama-server streams reasoning as `delta.reasoning_content`, a channel separate from
    # `delta.content` — NOT inline `<think>` tags in `text` (those only show up if a build/
    # template doesn't split the two). Captured here rather than left to fall on the floor,
    # for the one caller that turns reasoning on (`runner.run_thinking_experiment`); empty for
    # every normal call, where reasoning stays off at launch and this channel never fires.
    reasoning_text: str = ""

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling instead of stopping. On the plan task this is a real and
        common failure mode (a whole 8-semester schedule is a lot of JSON), so it is recorded
        rather than left to look like a malformed response."""
        return self.finish_reason == "length"


class LlamaCppClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0,
                 search_provider: Any = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # A callable `(query) -> str`, or None. See `_dispatch_tool` for why none ships here.
        self.search_provider = search_provider

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
        except OSError as exc:
            # NOT a urllib.error.URLError: a reset/broken-pipe/timeout that happens AFTER the
            # TCP connect succeeds (`ConnectionResetError`, `BrokenPipeError`, socket-level
            # `TimeoutError`) raises the raw `OSError` subclass, not `URLError` — urlopen only
            # wraps failures during the connect itself. Uncaught, this crashed a whole sweep
            # (2026-08-04): `server.start_server`'s health-check poll hit a `ConnectionReset`
            # while llama-server was still binding its port, and the exception propagated all
            # the way out of `run_convergence`, killing the harness process mid-run and
            # orphaning the GPU-resident server it had just launched. `health()`/`props()`
            # already treat `LlamaCppError` as "not ready yet, keep polling" — this makes a
            # startup-window connection hiccup one of those instead of a fatal crash.
            raise LlamaCppError(
                f"{method} {path} -> connection to {self.base_url} reset or failed ({exc})."
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

    # --- tools ---------------------------------------------------------------------------
    #
    # WHY THE QA/EXPLAIN STAGES GET TOOLS AND THE PLAN STAGES DO NOT. A plan is built from one
    # program's requirement rows, and the harness already knows exactly which rows those are —
    # there is nothing for the model to go looking for, and a tool call would only add round
    # trips to a stage whose latency is already the complaint. An advising QUESTION is the
    # opposite: what the answer needs depends on what was asked, which is precisely the case
    # where letting the model fetch beats guessing what to stuff in the prompt.
    #
    # THE SEARCH BACKEND IS NOT INCLUDED, and that is deliberate rather than unfinished. The
    # obvious free endpoints (DuckDuckGo's HTML view, Bing's) disallow automated clients in
    # robots.txt, which is the same line this project already declined to cross for Banner
    # prerequisites. So `web_search` dispatches to whatever provider `config.yaml`'s
    # `tools.web_search` names, and with none configured it returns a plain "search is
    # unavailable" result — the model is told, in band, that the tool did not work, which is
    # exactly what it would need to hear if a real provider timed out. Add a key and the
    # capability is live with no code change.
    WEB_SEARCH_TOOL: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information this system does not hold — "
                "university policy pages, deadlines, department announcements. Use it when "
                "the provided context does not answer the question. Do not use it for the "
                "student's own record, which is private and not on the web."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    }

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name != "web_search":
            return f"error: no tool named {name!r}"
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: web_search needs a non-empty query"
        return self.search_provider(query) if self.search_provider else (
            "Search is unavailable: no provider is configured for this deployment. "
            "Answer from the context you were given, and say plainly if it is not enough."
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        options: dict[str, Any],
        think: bool | None = None,
        response_format: dict | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tool_calls: int = 3,
    ) -> GenerationResult:
        """One streamed chat completion. Same message shape the app sends.

        `tools`, when given, turns this into a bounded tool loop: the model may call a tool,
        read its result and answer, up to `max_tool_calls` times. Streaming is kept for the
        final answer so TTFT still means what it means everywhere else in this harness; the
        tool-call turns are not streamed, because a tool call is not something a student is
        watching arrive.
        """
        start_tools = time.perf_counter()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_calls_made: list[dict[str, Any]] = []
        if tools:
            messages, tool_calls_made, direct = self._run_tool_loop(
                messages, tools, options, max_tool_calls)
            # NO SECOND GENERATION WHEN NO TOOL WAS CALLED. The tool loop's first turn already
            # produced a complete answer in that case, and re-asking for it streamed cost a
            # whole extra generation per question — measured on gemma4-e4b's 39-item QA bank,
            # median 1.1 s -> 7.6 s purely from offering tools nobody used. TTFT is unavailable
            # for this path (the answer did not arrive as a stream) and is reported as None
            # rather than as a fabricated number.
            if direct is not None:
                return GenerationResult(
                    text=direct.get("text", "").strip(),
                    ttft_s=None,
                    total_s=time.perf_counter() - start_tools,
                    eval_count=direct.get("completion_tokens"),
                    prompt_eval_count=direct.get("prompt_tokens"),
                    prompt_ms=None,
                    predicted_ms=None,
                    finish_reason=direct.get("finish_reason"),
                    raw_final={"usage": direct.get("usage") or {}, "timings": {},
                               "tool_calls": tool_calls_made},
                    reasoning_text=direct.get("reasoning", "").strip(),
                )
        payload: dict[str, Any] = {
            "messages": messages,
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
        #
        # `think is True` is the mirror case, used only by the server the thinking experiment
        # launches with reasoning enabled (see server.build_argv's `thinking_budget`) — this is
        # the request-level ask that goes with that server-level allowance, not an override of
        # a `--reasoning off` launch (which per llama.cpp's own budget=0 behavior a per-request
        # kwarg cannot undo).
        kwargs = dict(chat_template_kwargs or {})
        if think is False:
            kwargs.setdefault("enable_thinking", False)
        elif think is True:
            kwargs.setdefault("enable_thinking", True)
        if kwargs:
            payload["chat_template_kwargs"] = kwargs

        start = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        reasoning_parts: list[str] = []
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
                    delta = choice.get("delta") or {}
                    reasoning_piece = delta.get("reasoning_content") or ""
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                    piece = delta.get("content") or ""
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
            raw_final={"usage": usage, "timings": timings,
                       "tool_calls": tool_calls_made},
            reasoning_text="".join(reasoning_parts).strip(),
        )

    def _run_tool_loop(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        options: dict[str, Any], max_tool_calls: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        """Let the model call tools until it stops asking or hits the cap.

        NON-STREAMING, and bounded twice over: `max_tool_calls` turns, and every tool result is
        appended as a normal `tool` message so the final streamed answer is generated from a
        conversation the model can see in full. A model that loops asking for the same thing
        runs out of turns and then answers with whatever it has, which is the behaviour a
        student-facing deployment wants — never an unbounded agent.
        """
        made: list[dict[str, Any]] = []
        for turn in range(max_tool_calls):
            payload = {
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
                **{k: v for k, v in options.items() if k != "max_tokens"},
                "max_tokens": options.get("max_tokens"),
            }
            with self._request("POST", "/v1/chat/completions", payload) as resp:
                body = json.loads(resp.read().decode(errors="replace"))
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            if not calls:
                # Answered outright. Hand the text back so the caller can skip the streamed
                # turn entirely — see `chat`'s `direct` branch.
                usage = body.get("usage") or {}
                return messages, made, {
                    "text": message.get("content") or "",
                    "reasoning": message.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "usage": usage,
                }
            messages.append({k: v for k, v in message.items() if k in
                             ("role", "content", "tool_calls")})
            for call in calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self._dispatch_tool(name, arguments)
                made.append({"name": name, "arguments": arguments,
                             "result_chars": len(result)})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": result,
                })
        return messages, made, None
