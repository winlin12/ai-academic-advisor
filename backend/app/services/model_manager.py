"""Owns the vLLM server's process lifecycle, so a user can choose which local model answers
them.

WHY THIS EXISTS. Before it, the inference server was always started by hand outside the app —
`VllmClient` is a pure HTTP client that talks to whatever is listening at
`settings.vllm_base_url` and has no opinion about what model that is. That was fine when there
was exactly one model and switching meant a person on the box killing and relaunching a
process. It stops being fine the moment "pick your model" is a button in the product:
something inside the request path has to actually own starting, stopping and relaunching the
server. This module is that something.

ONE MODEL RESIDENT AT A TIME, ON PURPOSE. The box has 24GB of VRAM on one RTX 3090 Ti, and a
single 26-27B-class checkpoint at int4 uses 17-19.5GB of it — there is no room to keep two
loaded simultaneously, and the KV cache that is left over is what the app's concurrency is
made of. So switching genuinely means: stop the running server, wait for it to release VRAM,
start the new one, wait for it to report healthy. That is a real wait — LONGER than under
llama.cpp, not shorter, because vLLM also profiles a forward pass and captures CUDA graphs
before it serves — and the API and the UI both need to say so rather than pretend otherwise.

WHAT THE SWITCH TO vLLM CHANGED HERE, 2026-08-22:

  * THE FLAGS ARE GONE. Every `ModelSpec` used to carry `tensor_split` and `n_cpu_moe`, copied
    verbatim from model_eval so a model behaved in production exactly as it was measured.
    Those existed to fit a model across two mismatched cards. On one card there is nothing to
    split and nothing to offload, so a spec is now just an identity: a name, a label, a
    checkpoint directory, and how it is expected to behave.
  * WAITING FOR VRAM IS NOW EXPLICIT. vLLM's API server parents engine-core worker processes,
    and the CUDA context is released when THOSE exit, not when the parent does. Starting the
    next model too early means it sizes its KV pool against a card that is still occupied —
    which does not error, it just silently caps concurrency until the next restart.
  * CONCURRENCY IS A SETTING NOW. `--max-num-seqs` (settings.vllm_max_concurrent_seqs) is
    passed at launch. Under llama.cpp the equivalent stole context from every request and was
    pinned at 1; here it costs only KV.

SAME PORT, ALWAYS. `settings.vllm_base_url` never changes across a switch — only the checkpoint
does. That is what lets every existing call site (`VllmClient()`, instantiated fresh in half a
dozen routers) keep working with zero changes: they were already "whatever is answering on
8080", and that remains true.

REQUESTS DURING A SWITCH are not specially guarded here, and that is deliberate rather than an
oversight: the old server is stopped before the new one starts, so a request landing in that
window gets a connection refusal — which `VllmClient` already turns into
`VllmConnectionError`, and every caller already has a "model unavailable" fallback path for
that (the deterministic planner, a plain chat error). Adding a second, bespoke "switching"
guard in front of every route would duplicate a degradation path the app already has to have
anyway for the ordinary case of the server just being down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    # Stable id used in the API and by the frontend, and also the `--served-model-name` the
    # server is launched with, so /v1/models reports exactly this string back. Never a
    # filesystem path, so moving or requantizing a checkpoint doesn't break a client that has
    # this cached.
    name: str
    label: str
    # Checkpoint DIRECTORY under `vllm_models_root` — config.json plus safetensors shards plus
    # tokenizer. Not a single file the way the gguf was.
    model_path: str
    blurb: str
    # Overrides `settings.vllm_kv_cache_dtype` for this model only. NOT a tuning preference:
    # whether fp8 KV is available depends on which attention backend the model's shape selects,
    # and on this card the Triton backend rejects it outright at engine start (see gemma4-26b
    # below). None = use the global setting.
    kv_cache_dtype: str | None = None


# The two models model_eval's sweeps measure head-to-head — see model_eval/config.yaml, which
# these entries mirror, so a model selected here behaves exactly as the eval measured it.
#
# DOWN FROM FIVE, 2026-08-22. The three that went (qwen3.5-9b, gemma4-12b, qwen3.6-35b-a3b)
# were dropped rather than requantized: the small two existed to fit the smaller of two cards,
# a distinction that stopped existing when both cards were replaced by one, and the 35B-A3B was
# the previous generation of the MoE that remains. Their GGUFs are still on disk and their
# results are still in model_eval/results_full_3060/.
AVAILABLE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="gemma4-26b",
        label="Gemma 4 26B",
        model_path="gemma4-26b-awq",
        blurb="Fastest to answer — the default.",
        # fp16 KV, not the fp8 the other model uses. This model's shape puts it on vLLM's
        # TRITON attention backend, which refuses an fp8 KV cache on compute capability 8.6:
        #
        #   ValueError: FP8 KV cache is not supported by the Triton attention backend on
        #   NVIDIA GeForce RTX 3090 Ti (compute capability 8.6)
        #
        # It fails at engine start, not at request time — so with the global fp8 setting the
        # DEFAULT model never comes up, and (stderr going to DEVNULL) the only symptom is a
        # switch that times out. Mirrors model_eval/config.yaml's per-model override.
        kv_cache_dtype="auto",
    ),
    ModelSpec(
        name="qwen3.8-27b",
        label="Qwen 3.8 27B",
        model_path="qwen3.8-27b-awq",
        blurb="Higher plan quality; slower per answer.",
    ),
)
BY_NAME: dict[str, ModelSpec] = {m.name: m for m in AVAILABLE_MODELS}
# Gemma 4 26B is the default because it is far faster to answer. The pre-2026-08-22 note here
# said the decode bottleneck was the RTX 2060 Super itself (~17-18 tok/s regardless of tensor
# split) — a hardware ceiling that no longer exists. MEASURED 2026-08-22 on the 3090 Ti, both
# at 16384 context:
#
#   gemma4-26b   ~147-202 tok/s decode   56,117 KV tokens   3.43x concurrency
#   qwen3.8-27b  ~48 tok/s decode        77,101 KV tokens   4.71x concurrency
#
# NOTE WHICH WAY THE CONCURRENCY GOES. Gemma is the faster model but the tighter one on seats,
# because it is forced onto fp16 KV (see its `kv_cache_dtype` above) and its full-attention
# layers use a 512-wide head against Qwen's 256. If the deployment ever becomes seat-limited
# rather than latency-limited, that is the tradeoff to revisit — and it is the opposite of what
# the layer counts suggest.
DEFAULT_MODEL = "gemma4-26b"


class ModelSwitchError(RuntimeError):
    """A switch was requested but the new process never became healthy."""


class UnknownModelError(ValueError):
    def __init__(self, name: str):
        super().__init__(
            f"unknown model {name!r} — available: {', '.join(BY_NAME)}"
        )


def _build_argv(settings: Settings, spec: ModelSpec, port: int) -> list[str]:
    """The production launch line. Deliberately the same shape as model_eval's `build_argv`.

    KEEP THESE TWO IN STEP. The eval's claim to measure the shipped path is only true while the
    flags match — that claim was already falsified once on exactly this axis (2026-07-29: the
    eval ran Mode A at 16384 while production ran a 1024-token output budget, and silently
    broke revise-plan). The settings that must agree are the context length, the KV dtype and
    the memory fraction; `--max-num-seqs` deliberately does NOT, because the eval measures one
    student's latency serially and production serves several at once.
    """
    return [
        settings.vllm_server_exe, "serve",
        str(Path(settings.vllm_models_root) / spec.model_path),
        "--served-model-name", spec.name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-model-len", str(settings.vllm_context_tokens),
        "--max-num-seqs", str(settings.vllm_max_concurrent_seqs),
        "--gpu-memory-utilization", str(settings.vllm_gpu_memory_utilization),
        "--kv-cache-dtype", spec.kv_cache_dtype or settings.vllm_kv_cache_dtype,
        # TEXT-ONLY, DECLARED. Both checkpoints are multimodal and this app never sends an
        # image or an audio clip. Left at its default, vLLM reserves multimodal preprocessor
        # memory for a modality that will never arrive and takes it straight out of the KV
        # pool — i.e. out of how many students can be served at once.
        "--limit-mm-per-prompt", '{"image":0,"video":0,"audio":0}',
        # Every request opens with the same system prompt and the same rendered requirement
        # export; without this each one is re-prefilled from scratch.
        "--enable-prefix-caching",
    ]


def _server_env(settings: Settings) -> dict[str, str]:
    """This process's environment, plus the venv's `bin/` on PATH.

    vLLM SHELLS OUT to build tools (`ninja`) during engine startup, and those are console
    scripts installed alongside `vllm` in the same `bin/`. The venv is not activated here — the
    server is invoked by absolute path — so without this the launch dies minutes in with a
    `FileNotFoundError: 'ninja'` buried in a long traceback, which surfaces to the user as
    "that model failed to load" rather than as the PATH problem it is.

    Doubly worth guarding in production: `stderr` is sent to DEVNULL below, so the traceback
    that explains it would not be written down anywhere at all.
    """
    env = dict(os.environ)
    bin_dir = str(Path(settings.vllm_server_exe).resolve().parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


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
        # vllm_base_url is always http://host:port with no path — see config.py's comment
        # on why 127.0.0.1 specifically. Parsed rather than duplicated as a separate setting,
        # so the two can never quietly disagree.
        return httpx.URL(self.settings.vllm_base_url).port or 8080

    async def _health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.settings.vllm_base_url}/health")
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
            logger.warning("vLLM did not exit within 30s of SIGTERM; sending SIGKILL")
            process.kill()
            await process.wait()
        # THE PARENT EXITING IS NOT THE VRAM COMING BACK. vLLM's API server parents engine-core
        # worker processes and it is those that hold the CUDA context, so returning here the
        # moment `process.wait()` resolves lets the NEXT model start against a card that is
        # still several GB occupied. It does not fail — it sizes a smaller KV pool and quietly
        # serves fewer students until something restarts it, which is the kind of bug that
        # gets diagnosed as "the app got slow" weeks later.
        await self._await_vram_release()

    @staticmethod
    async def _await_vram_release(settle_mb: int = 1024, timeout_s: float = 120.0) -> None:
        """Poll nvidia-smi until the card is back to roughly idle, or give up after `timeout_s`.

        Giving up is NOT an error: something else on the box may legitimately hold VRAM, and
        refusing to start a model in that case would be worse than starting a squeezed one.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await proc.communicate()
            except (OSError, ValueError):
                return  # no nvidia-smi to ask; nothing to wait for
            try:
                if max(int(x) for x in out.split()) < settle_mb:
                    return
            except ValueError:
                return
            await asyncio.sleep(2.0)
        logger.warning(
            "model_manager: VRAM still in use %.0fs after the server exited; starting the next "
            "model anyway — its KV cache may be undersized", timeout_s)

    async def switch(self, name: str) -> None:
        """Stop whatever is running, start `name`, wait for it to answer /health.

        Raises `UnknownModelError` for a bad name, `ModelSwitchError` if the new process never
        becomes healthy within `vllm_startup_timeout_s` — in both cases nothing is left
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
                logger.info("model_manager: launching %s (%s)", name, spec.model_path)
                self._process = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=_server_env(self.settings),
                )
                healthy = await self._wait_healthy(self.settings.vllm_startup_timeout_s)
                if not healthy:
                    await self._stop_process()
                    self.current = None
                    self.last_error = (
                        f"{spec.label} did not report healthy within "
                        f"{self.settings.vllm_startup_timeout_s:.0f}s"
                    )
                    raise ModelSwitchError(self.last_error)
                self.current = name
                # `VllmClient()` (instantiated fresh in every router, no args) defaults its
                # `model` to `settings.vllm_model` — updating it here is what keeps every one
                # of those call sites, and the health-check mismatch probe in
                # vllm_client.health(), describing the model actually running rather than
                # whatever `.env` said at process start. Without this, logs and health checks
                # would keep claiming the OLD model's name after every successful switch.
                #
                # THE SERVED NAME, not a filename — that is what `--served-model-name` puts on
                # /v1/models, and `health()` now compares the two exactly rather than reducing
                # both to a basename the way the gguf path forced it to.
                self.settings.vllm_model = spec.name
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
