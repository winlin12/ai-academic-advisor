"""Owns llama-server's process lifecycle, so a user can choose which local model answers them.

WHY THIS EXISTS. Before it, llama-server was always started by hand outside the app —
`LlamaCppClient` is a pure HTTP client that talks to whatever is listening at
`settings.llamacpp_base_url` and has no opinion about what model that is. That was fine when
there was exactly one model and switching meant a person on the box killing and relaunching a
process. It stops being fine the moment "pick your model" is a button in the product: something
inside the request path has to actually own starting, stopping and relaunching llama-server.
This module is that something.

ONE MODEL RESIDENT AT A TIME, ON PURPOSE. The box has 20GB of VRAM across two cards (RTX 3060
12GB + RTX 2060 Super 8GB) and a single 26-35B-class model at Q4_K_M already uses most of
that — there is no room to keep two loaded simultaneously without heavy CPU offload, which
would make whichever model isn't "hot" noticeably slower to answer. So switching genuinely
means: stop the running server, wait for it to release VRAM, start the new one, wait for it to
report healthy. That is a real 30-90 second wait, not an instant toggle, and the API and the UI
both need to say so rather than pretend otherwise.

SAME PORT, ALWAYS. `settings.llamacpp_base_url` never changes across a switch — only the `-m`
gguf and its per-model launch flags (tensor-split, CPU-offloaded MoE layers) do. That is what
lets every existing call site (`LlamaCppClient()`, instantiated fresh in half a dozen routers)
keep working with zero changes: they were already "whatever is answering on 8080", and that
remains true.

REQUESTS DURING A SWITCH are not specially guarded here, and that is deliberate rather than an
oversight: the old server is stopped before the new one starts, so a request landing in that
window gets a connection refusal — which `LlamaCppClient` already turns into
`LlamaCppConnectionError`, and every caller already has a "model unavailable" fallback path for
that (the deterministic planner, a plain chat error). Adding a second, bespoke "switching"
guard in front of every route would duplicate a degradation path the app already has to have
anyway for the ordinary case of llama-server just being down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    # Stable id used in the API and by the frontend — never the raw gguf filename, so a file
    # rename doesn't break a client that has this cached.
    name: str
    label: str
    gguf: str
    blurb: str
    tensor_split: str | None = None
    # CPU-offloaded MoE expert layers — only qwen3.6-35b-a3b needs this to fit in 20GB VRAM at
    # all; see model_eval/config.yaml, which measured these exact settings.
    n_cpu_moe: int | None = None


# The five models model_eval's sweeps measured head-to-head — see model_eval/config.yaml,
# which these flags are copied from verbatim (same tensor-split, same n_cpu_moe) so a model
# selected here behaves exactly as the eval measured it, not as a re-guessed approximation.
AVAILABLE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="qwen3.5-9b",
        label="Qwen 3.5 9B",
        gguf="qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf",
        tensor_split="1,0",
        blurb="Smallest and fastest; lowest plan quality of the five.",
    ),
    ModelSpec(
        name="qwen3.6-27b",
        label="Qwen 3.6 27B",
        gguf="qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf",
        # 0.63,0.37 — MUST match model_eval/config.yaml exactly (it drifted to 0.67,0.33 here at
        # some point; fixed 2026-08-07). The eval's own tensor-split benchmark that day measured
        # 0.63/0.37 as the fastest working split for this model and 0.67-ish already noticeably
        # slower, so the drift was not just an inconsistency, it was actively worse.
        tensor_split="0.63,0.37",
        blurb="Best plan quality measured in the eval; slower to answer than Gemma.",
    ),
    ModelSpec(
        name="gemma4-12b",
        label="Gemma 4 12B",
        gguf="gemma4-12b/gemma-4-12b-it-IQ4_XS.gguf",
        tensor_split="1,0",
        blurb="Small and fast; plan quality below the 26B/27B pair.",
    ),
    ModelSpec(
        name="gemma4-26b",
        label="Gemma 4 26B",
        gguf="gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        tensor_split="0.63,0.37",
        blurb="Fastest of the large models — the default.",
    ),
    ModelSpec(
        name="qwen3.6-35b-a3b",
        label="Qwen 3.6 35B (A3B)",
        gguf="qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        tensor_split="0.67,0.33",
        n_cpu_moe=9,
        blurb="Largest model; partly runs on CPU, noticeably slower to answer.",
    ),
)
BY_NAME: dict[str, ModelSpec] = {m.name: m for m in AVAILABLE_MODELS}
# Gemma 4 26B, not Qwen — changed back 2026-08-07. Qwen 3.6 27B measured better plan quality,
# but model_eval's own tensor-split benchmark that day found the decode bottleneck is the RTX
# 2060 Super itself (~17-18 tok/s regardless of split ratio, and shifting more of the model
# onto the OTHER card made it slower, not faster) — a hardware ceiling, not something either
# model's launch flags can tune away. Gemma is the faster of the two at that same ceiling, so
# it is the default a first-time visitor gets; Qwen stays one switch away (see AVAILABLE_MODELS
# above) for whoever wants to spend the extra time for its better plan quality.
DEFAULT_MODEL = "gemma4-26b"


class ModelSwitchError(RuntimeError):
    """A switch was requested but the new process never became healthy."""


class UnknownModelError(ValueError):
    def __init__(self, name: str):
        super().__init__(
            f"unknown model {name!r} — available: {', '.join(BY_NAME)}"
        )


def _build_argv(settings: Settings, spec: ModelSpec, port: int) -> list[str]:
    gguf_path = Path(settings.llamacpp_models_root) / spec.gguf
    argv = [
        settings.llamacpp_server_exe,
        "-m", str(gguf_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(settings.llamacpp_context_tokens),
        "--n-gpu-layers", "999",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--parallel", "1",
        "--reasoning", "off",
        "--reasoning-budget", "0",
    ]
    if spec.tensor_split:
        argv += ["--tensor-split", spec.tensor_split]
    if spec.n_cpu_moe is not None:
        argv += ["--n-cpu-moe", str(spec.n_cpu_moe)]
    return argv


@dataclass
class ModelManager:
    settings: Settings
    current: str | None = None
    # Set the instant a switch is requested, cleared when it settles (success or failure) — the
    # status endpoint reads this so the picker can show "switching to X" rather than nothing.
    switching_to: str | None = None
    last_error: str | None = None
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def _port(self) -> int:
        # llamacpp_base_url is always http://host:port with no path — see config.py's comment
        # on why 127.0.0.1 specifically. Parsed rather than duplicated as a separate setting,
        # so the two can never quietly disagree.
        return httpx.URL(self.settings.llamacpp_base_url).port or 8080

    async def _health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.settings.llamacpp_base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _wait_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await self._health():
                return True
            if self._process is not None and self._process.returncode is not None:
                # Exited on its own (bad gguf path, OOM, port already in use) — no point
                # polling out the rest of the timeout for a process that is already gone.
                return False
            await asyncio.sleep(1.0)
        return False

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("llama-server did not exit within 30s of SIGTERM; sending SIGKILL")
            process.kill()
            await process.wait()

    async def switch(self, name: str) -> None:
        """Stop whatever is running, start `name`, wait for it to answer /health.

        Raises `UnknownModelError` for a bad name, `ModelSwitchError` if the new process never
        becomes healthy within `llamacpp_startup_timeout_s` — in both cases nothing is left
        running under the OLD model's identity, but the caller should treat "no model loaded"
        as a real possibility until the next successful switch.
        """
        if name not in BY_NAME:
            raise UnknownModelError(name)
        spec = BY_NAME[name]
        async with self._lock:
            self.switching_to = name
            self.last_error = None
            try:
                await self._stop_process()
                argv = _build_argv(self.settings, spec, self._port())
                logger.info("model_manager: launching %s (%s)", name, spec.gguf)
                self._process = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                healthy = await self._wait_healthy(self.settings.llamacpp_startup_timeout_s)
                if not healthy:
                    await self._stop_process()
                    self.current = None
                    self.last_error = (
                        f"{spec.label} did not report healthy within "
                        f"{self.settings.llamacpp_startup_timeout_s:.0f}s"
                    )
                    raise ModelSwitchError(self.last_error)
                self.current = name
                # `LlamaCppClient()` (instantiated fresh in every router, no args) defaults its
                # `model` to `settings.llamacpp_model` — updating it here is what keeps every
                # one of those call sites, and the health-check mismatch probe in
                # llamacpp_client.health(), describing the model actually running rather than
                # whatever `.env` said at process start. Without this, logs and health checks
                # would keep claiming the OLD model's name after every successful switch.
                self.settings.llamacpp_model = Path(spec.gguf).name
            finally:
                self.switching_to = None

    async def start_default(self) -> None:
        """Best-effort at app startup: failing to launch a model must not stop the API from
        serving everything else (catalog browsing, the deterministic planner) — the same
        "degrade, don't crash" rule `main.py`'s lifespan already applies to the RAG schema."""
        try:
            await self.switch(DEFAULT_MODEL)
        except (ModelSwitchError, UnknownModelError, OSError) as exc:
            logger.warning("model_manager: startup launch of %s failed: %s", DEFAULT_MODEL, exc)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_process()
            self.current = None

    def status(self) -> dict:
        return {
            "current": self.current,
            "switching_to": self.switching_to,
            "last_error": self.last_error,
            "available": [
                {"name": m.name, "label": m.label, "blurb": m.blurb}
                for m in AVAILABLE_MODELS
            ],
        }
