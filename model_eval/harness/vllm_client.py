"""Minimal vLLM OpenAI-server HTTP client (stdlib only — the harness must not depend on the app).

Replaces the llama.cpp client. The wire format barely moved — both speak OpenAI
``/v1/chat/completions`` and both stream — so what changed is the three places llama.cpp had
its own dialect:

1. TIMINGS ARE GONE, AND THAT IS FINE. vLLM server answered with its own ``prompt_ms`` /
   ``predicted_ms`` counters when asked for ``timings_per_token``; vLLM sends no such block,
   and its request model rejects unknown fields rather than ignoring them, so the flag is not
   merely useless here, it is a 400. ``prompt_ms``/``predicted_ms`` are therefore derived from
   the WALL CLOCK: prefill is time-to-first-token, decode is everything after it. That is a
   deliberate downgrade in precision and an upgrade in relevance — a student waits on the wall
   clock, not on the engine's self-report — but it does mean throughput recorded here is not
   comparable token-for-token with the llama.cpp-era records. ``raw_final["timings_source"]``
   says which of the two produced any given row.

2. THERE IS NO ``/props``. llama.cpp exposed the launch settings for read-back, which is how
   the harness proved every model really ran at ``num_ctx``. vLLM puts the same fact on
   ``/v1/models`` as ``max_model_len``, so ``server_info()`` reads it from there and
   ``runner`` records it exactly as before. ``/health`` also differs: it answers 200 with an
   EMPTY body, so status code is the signal and parsing it as JSON would fail every poll.

3. CONTEXT IS NO LONGER DIVIDED BY THE SLOT COUNT. Under llama.cpp ``--parallel N`` cut the
   per-request window to ``--ctx-size / N``, which is why the harness carried
   ``slot_context()`` and refused to run when a prompt would not fit a slot. vLLM's
   PagedAttention allocates KV by the block as sequences actually grow, so ``--max-model-len``
   is what EVERY concurrent request gets and raising ``--max-num-seqs`` costs no context at
   all. This is the reason the project moved; see ``server.py``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class VllmError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    ttft_s: float | None            # wall-clock, request sent -> first content token
    total_s: float                  # wall-clock, request sent -> stream closed
    eval_count: int | None          # completion tokens
    prompt_eval_count: int | None   # prompt tokens
    # WALL-CLOCK DERIVED, not engine-reported — see the module docstring. `prompt_ms` is
    # time-to-first-token (prefill plus queueing plus HTTP), `predicted_ms` is everything
    # after it (decode). Both are None when there was no stream to time, which is the
    # tool-loop's direct-answer path.
    prompt_ms: float | None
    predicted_ms: float | None
    finish_reason: str | None = None
    raw_final: dict[str, Any] = field(default_factory=dict)
    # vLLM streams reasoning as `delta.reasoning` (llama.cpp used `reasoning_content`; the
    # client reads both), a channel separate from
    # `delta.content` — NOT inline `<think>` tags in `text` (those only show up if a build/
    # template doesn't split the two). Captured here rather than left to fall on the floor,
    # for the callers that turn reasoning on (`run.advising_thinking`); empty for every normal
    # call, where the template is asked for `enable_thinking: false` and this never fires.
    reasoning_text: str = ""

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling instead of stopping. On the plan task this is a real and
        common failure mode (a whole 8-semester schedule is a lot of JSON), so it is recorded
        rather than left to look like a malformed response."""
        return self.finish_reason == "length"


class VllmClient:
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
            raise VllmError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise VllmError(
                f"Cannot reach vLLM server at {self.base_url} ({exc.reason})."
            ) from exc
        except OSError as exc:
            # NOT a urllib.error.URLError: a reset/broken-pipe/timeout that happens AFTER the
            # TCP connect succeeds (`ConnectionResetError`, `BrokenPipeError`, socket-level
            # `TimeoutError`) raises the raw `OSError` subclass, not `URLError` — urlopen only
            # wraps failures during the connect itself. Uncaught, this crashed a whole sweep
            # (2026-08-04): `server.start_server`'s health-check poll hit a `ConnectionReset`
            # while vLLM server was still binding its port, and the exception propagated all
            # the way out of `run_convergence`, killing the harness process mid-run and
            # orphaning the GPU-resident server it had just launched. `health()`/`props()`
            # already treat `VllmError` as "not ready yet, keep polling" — this makes a
            # startup-window connection hiccup one of those instead of a fatal crash.
            raise VllmError(
                f"{method} {path} -> connection to {self.base_url} reset or failed ({exc})."
            ) from exc

    # --- probes ---------------------------------------------------------------------------

    def health(self, timeout: float = 5.0) -> bool:
        """STATUS CODE ONLY. vLLM's /health answers 200 with an EMPTY body; llama.cpp's
        answered `{"status": "ok"}`. Parsing this as JSON fails on every poll, which reads as
        "server never came up" and times out a launch that in fact succeeded minutes earlier.
        """
        try:
            with self._request("GET", "/health", timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except VllmError:
            return False

    def props(self) -> dict[str, Any]:
        """Server-side truth about what is actually loaded: context size and model id.

        THERE IS NO /props ON vLLM. The fact the harness needs from it — "is every model
        really running at num_ctx?" — is on /v1/models as `max_model_len`, so this reads it
        from there and re-shapes it into the `{"default_generation_settings": {"n_ctx": ...}}`
        envelope `runner` already destructures. Keeping the adapter here rather than teaching
        the runner two shapes means exactly one place knows the endpoint moved.
        """
        # llama.cpp STILL HAS /props AND IT IS THE BETTER ANSWER, so try it first. It reports
        # the context the server actually allocated per slot, which is the number this check
        # exists to verify; vLLM has no such endpoint and 404s here, costing one cheap request
        # before the /v1/models path below. Ordering it this way keeps both engines on one code
        # path instead of teaching the runner which server it is talking to.
        try:
            with self._request("GET", "/props", timeout=10) as resp:
                props = json.load(resp)
            settings = props.get("default_generation_settings") or {}
            if settings.get("n_ctx"):
                return {
                    "default_generation_settings": {"n_ctx": settings["n_ctx"]},
                    "model_path": props.get("model_path") or self.loaded_model(),
                    "max_model_len": settings["n_ctx"],
                }
        except (VllmError, json.JSONDecodeError, AttributeError):
            pass

        entry = self._model_entry()
        if not entry:
            return {}
        n_ctx = entry.get("max_model_len")
        return {
            "default_generation_settings": {"n_ctx": n_ctx},
            "model_path": entry.get("id"),
            "max_model_len": n_ctx,
        }

    def _model_entry(self) -> dict[str, Any]:
        try:
            with self._request("GET", "/v1/models", timeout=10) as resp:
                data = json.load(resp)
        except (VllmError, json.JSONDecodeError):
            return {}
        entries = data.get("data") or []
        return entries[0] if entries else {}

    def loaded_model(self) -> str | None:
        return self._model_entry().get("id")

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
                    raw_final={"usage": direct.get("usage") or {},
                               "timings_source": "wall_clock",
                               "tool_calls": tool_calls_made},
                    reasoning_text=direct.get("reasoning", "").strip(),
                )
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            # NO `timings_per_token`. That was llama.cpp's opt-in for its own prompt_ms/
            # predicted_ms counters; vLLM's request model REJECTS unknown fields outright, so
            # sending it is a 400 on every generation rather than a harmless no-op.
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
        # asks a stage to reason — on vLLM that is purely a per-request matter, since there is
        # no launch-time reasoning switch to override. Its only user now is
        # `run.advising_thinking`; the `plan_b_thinking` arm was removed 2026-08-24.
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
                    raise VllmError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # BOTH SPELLINGS, AND vLLM USES THE SECOND. llama-server streamed the
                    # reasoning channel as `delta.reasoning_content`; vLLM 0.27.1's
                    # `--reasoning-parser` streams it as `delta.reasoning`. Reading only the
                    # llama.cpp name silently dropped EVERY reasoning token — VERIFIED
                    # 2026-08-23 by dumping the raw delta keys, which were
                    # `{'role': 1, 'content': 1, 'reasoning': 800}`.
                    #
                    # The failure was invisible in exactly the way that matters: a thinking run
                    # recorded `reasoning_chars: 0` while its `eval_count` jumped from 315 to
                    # 1747 on the same prompt, so the model plainly HAD reasoned and the
                    # harness's one measure of whether it did said no. Any thinking-vs-not
                    # comparison built on that field would have concluded "the model ignored
                    # the instruction" from a client bug.
                    reasoning_piece = (delta.get("reasoning")
                                       or delta.get("reasoning_content") or "")
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                    piece = delta.get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        total_s = time.perf_counter() - start
        # SPLIT THE WALL CLOCK AT FIRST TOKEN. vLLM reports no timing block of its own, so
        # prefill is "until the first content token arrived" and decode is "everything after",
        # and the report's tok/s is `eval_count / predicted_ms`, exactly as before. What is
        # measured has widened — queue time and HTTP now land in `prompt_ms` where llama.cpp's
        # counter excluded them — which is the honest number for a server that is meant to hold
        # several students at once, but it is NOT the same measurement as the pre-2026-08-22
        # records. `timings_source` marks every row so the two can never be silently pooled.
        #
        # A response with no content token at all (an empty completion, or a refusal that
        # streamed only reasoning) leaves `ttft` None; attributing all of it to decode would
        # invent a throughput number out of a generation that produced nothing, so both fields
        # stay None and the report drops the row rather than counting it.
        return GenerationResult(
            text="".join(parts).strip(),
            ttft_s=ttft,
            total_s=total_s,
            eval_count=usage.get("completion_tokens"),
            prompt_eval_count=usage.get("prompt_tokens"),
            prompt_ms=ttft * 1000 if ttft is not None else None,
            predicted_ms=(total_s - ttft) * 1000 if ttft is not None else None,
            finish_reason=finish_reason,
            raw_final={"usage": usage, "timings_source": "wall_clock",
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
                    "reasoning": (message.get("reasoning")
                                  or message.get("reasoning_content") or ""),
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
