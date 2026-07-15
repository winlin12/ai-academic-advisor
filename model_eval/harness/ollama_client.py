"""Minimal Ollama HTTP client (stdlib only — the harness must not depend on the app).

Streaming is mandatory: time-to-first-token can only be observed on a stream. The final
stream chunk carries Ollama's own timing/token counters (nanoseconds), which we prefer
over wall-clock where both exist.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    ttft_s: float | None          # wall-clock, request sent -> first response token
    total_s: float                # wall-clock, request sent -> stream closed
    eval_count: int | None        # output tokens (Ollama's counter)
    prompt_eval_count: int | None # prompt tokens actually evaluated
    load_duration_s: float | None # >0.5s here on a measured run means warmup failed
    eval_duration_s: float | None
    raw_final: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, payload: dict | None = None):
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            raise OllamaError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url} ({exc.reason}). Is it running?"
            ) from exc

    def version(self) -> str:
        with self._request("GET", "/api/version") as resp:
            return json.load(resp).get("version", "unknown")

    def ps(self) -> list[dict[str, Any]]:
        """Loaded models. NOTE: size_vram/size is a MEMORY split; it is recorded as a
        fallback signal only — layer offload truth comes from the server log."""
        with self._request("GET", "/api/ps") as resp:
            return json.load(resp).get("models", [])

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any],
        think: bool | None = None,
        output_format: dict | str | None = None,
        keep_alive: str | int = "15m",
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options,
            "keep_alive": keep_alive,
        }
        if think is not None:
            payload["think"] = think
        if output_format is not None:
            payload["format"] = output_format

        start = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        final: dict[str, Any] = {}
        with self._request("POST", "/api/generate", payload) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise OllamaError(f"{model}: {chunk['error']}")
                piece = chunk.get("response", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(piece)
                if chunk.get("done"):
                    final = chunk
        total = time.perf_counter() - start

        def ns_to_s(key: str) -> float | None:
            v = final.get(key)
            return v / 1e9 if isinstance(v, (int, float)) else None

        return GenerationResult(
            text="".join(parts).strip(),
            ttft_s=ttft,
            total_s=total,
            eval_count=final.get("eval_count"),
            prompt_eval_count=final.get("prompt_eval_count"),
            load_duration_s=ns_to_s("load_duration"),
            eval_duration_s=ns_to_s("eval_duration"),
            raw_final={k: v for k, v in final.items() if k not in ("response", "context")},
        )

    def unload(self, model: str) -> None:
        """Evict a model so the next model's VRAM measurement starts from a clean card."""
        try:
            with self._request(
                "POST", "/api/generate", {"model": model, "prompt": "", "keep_alive": 0}
            ) as resp:
                resp.read()
        except OllamaError:
            pass  # already unloaded / never loaded — fine
