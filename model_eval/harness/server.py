"""vLLM OpenAI-server process lifecycle.

WHY THE HARNESS OWNS THE PROCESS. Unchanged from the llama.cpp era, and for the same reason:
one process serves exactly one model, and the settings that decide what the comparison means
— context length, KV dtype, how much of the card the engine may claim, how many sequences may
run at once — are all fixed at launch. If the harness merely *connected* to whatever server
happened to be running, the central promise ("every model saw identical settings") would
depend on whoever typed the last command line. So the harness starts and stops the server
itself, from ``config.yaml``, and records the resulting command line in ``meta_*.json``.

WHAT CHANGED WHEN vLLM CAME IN, 2026-08-22. The old backend was chosen when this box
had two mismatched, power-limited cards (RTX 3060 12 GB + RTX 2060 SUPER 8 GB) and the whole
game was fitting a model across them: ``--n-gpu-layers``, ``--n-cpu-moe``, ``--tensor-split``,
and a log parser that counted per-layer device assignments because a model silently running
half on the CPU posts terrible latency and identical quality. A single RTX 3090 Ti 24 GB
replaced both, and every one of those knobs became a no-op — the model either fits or it does
not. What is left over is bandwidth and headroom, which is exactly what the old stack could
not spend:

  * PAGED KV. llama.cpp pre-allocated ``--ctx-size`` and DIVIDED it between ``--parallel``
    slots, so simulating four students meant giving each of them a quarter of the window.
    MEASURED on this workload (see config.yaml's `parallel`), four-way batching came out ~18%
    slower in aggregate than serving one student at a time. vLLM allocates KV by the block as
    sequences actually grow, so ``--max-model-len`` is what EVERY concurrent request gets and
    raising ``--max-num-seqs`` costs no context at all. For a thing that is a web server at
    the end of the day, that is the difference that matters.
  * PREFIX CACHING. Every request this harness sends opens with the same system prompt and,
    within a program, the same rendered requirement database. vLLM reuses those blocks across
    requests instead of re-prefilling them per call.

WHAT WAS GIVEN UP, recorded so nobody rediscovers it as a surprise. The weights are now
W4A16 ``compressed-tensors`` rather than GGUF Q4_K_M, which means NO QUALITY NUMBER RECORDED
HERE POOLS WITH ANYTHING IN ``results_full_3060/``. That is a fresh series, not a continuation
— see README. Speculative decoding is gone with the llama.cpp draft-model plumbing; vLLM has
its own (``--speculative-config``) and it is deliberately not wired up, because the draft that
paid on the old hardware paid by amortizing weight reads that no longer dominate.

BOTH MODELS ARE MULTIMODAL CHECKPOINTS used text-only. The vision tower is in the weights
whether or not it is used (~1.5 GB of the Qwen checkpoint's 19.5 GB, and no int4 quant on the
Hub strips it), but the per-request multimodal PREPROCESSOR memory is avoidable, and
``--limit-mm-per-prompt`` is what avoids it. See ``build_argv``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vllm_client import VllmClient

# KV-CACHE CAPACITY FROM THE ENGINE'S OWN STARTUP LOG. This is the direct replacement for the
# llama.cpp-era offload check, and it answers the question that actually decides whether a run
# means anything on this stack: how many tokens of KV did the engine end up with after the
# weights were loaded?
#
# Under llama.cpp the failure to catch was "half the layers are secretly on the CPU". Here the
# model is always fully resident or the server does not start at all — but a checkpoint that
# eats more of the card than expected silently shrinks the KV pool instead, and a shrunken pool
# does not error. It caps concurrency, which is the one thing this whole migration was for. A
# run whose KV pool collapsed to one sequence's worth would post perfectly good quality numbers
# and a completely misleading throughput story.
#
#   GPU KV cache size: 87,024 tokens
#   Maximum concurrency for 16,384 tokens per request: 5.31x
#
_KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I)
_MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", re.I)


def gpu_memory_used_mb() -> list[int] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return [int(x) for x in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_total_mb() -> int | None:
    """Total VRAM on the first card, for sizing checks that run before any server starts.

    `run.py check` uses it to answer "will these weights leave room for a KV cache?" without
    spending the several minutes a real launch costs to find out.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def resolve_host(configured: str, _server_exe: str | Path | None = None) -> str:
    """The address the harness connects to. ``auto`` -> loopback.

    ONE TOPOLOGY NOW. The llama.cpp era carried a second one — a Windows CUDA build driven
    over WSL interop, which needed ``wslpath`` translation on every path, a bind on 0.0.0.0
    because WSL2's NAT put the server off this box's loopback, and ``taskkill.exe`` to stop it
    because SIGTERM reached only the Linux-side proxy. That machinery is deleted: the box is
    native Linux with a native CUDA toolchain, and the second topology existed solely because
    there was no CUDA build available in it.

    An explicit value is still honoured, for pointing the harness at a server on another host.
    """
    return configured if configured and configured != "auto" else "127.0.0.1"


@dataclass
class ServerHandle:
    process: subprocess.Popen
    log_path: Path
    argv: list[str]
    base_url: str
    started_at: float = field(default_factory=time.time)

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def kv_cache(self) -> dict[str, Any]:
        """What the engine actually had left for KV after loading the weights.

        Recorded per model into ``meta_*.json`` the way ``offload()`` used to be. ``blocks``
        is reported in TOKENS rather than vLLM's internal 16-token pages, because tokens is
        the unit every other number in this harness is already in (``num_ctx``,
        ``max_plan_tokens``, ``prompt_eval_count``).

        ``max_concurrency`` is the engine's own arithmetic — KV tokens divided by
        ``--max-model-len`` — and it is the honest ceiling on how many students this model can
        serve at once at full context. It is a FLOOR on real concurrency, not a cap: it assumes
        every sequence runs to the full window, and with prefix caching on, shared prompt
        blocks are counted once rather than per sequence.

        Not knowing is reported as not knowing, exactly as before — the report needs to be
        able to say which of the two it is looking at.
        """
        log = self.log_text()
        tokens = _KV_TOKENS_RE.findall(log)
        concurrency = _MAX_CONCURRENCY_RE.findall(log)
        if not tokens:
            return {"source": "unverified", "kv_cache_tokens": None, "max_concurrency": None}
        return {
            "source": "vllm_startup_log",
            "kv_cache_tokens": int(tokens[-1].replace(",", "")),
            "max_concurrency": float(concurrency[-1][1]) if concurrency else None,
            "per_request_tokens": (
                int(concurrency[-1][0].replace(",", "")) if concurrency else None),
        }

    def stop(self, timeout: float = 30.0, *, trim_log_lines: int | None = None) -> None:
        """SIGTERM, then SIGKILL, then wait for the VRAM to actually come back.

        THE WAIT IS NOT PARANOIA. vLLM's API server is a parent to engine-core worker
        processes, and the CUDA context those workers hold is released when they exit, not
        when the parent does. Starting the next model while the previous one's workers are
        still tearing down means its `--gpu-memory-utilization` is measured against a card that
        is still several GB occupied, and the engine sizes a too-small KV pool from it — which
        does not error, it just quietly caps concurrency for that model only. That is precisely
        the cross-model contamination this harness exists to prevent, so the handle does not
        report itself stopped until the card is actually back to baseline.
        """
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        self._await_vram_release()
        if trim_log_lines:
            self.trim_log(trim_log_lines)

    @staticmethod
    def _await_vram_release(settle_mb: int = 1024, timeout: float = 120.0) -> None:
        """Block until the card reports less than `settle_mb` in use, or `timeout` passes.

        A timeout is NOT raised. If something else on the box legitimately holds VRAM, failing
        the sweep here would be worse than proceeding — the KV-cache size that results is
        recorded in `meta_*.json` either way, so a squeezed run is visible in the report
        rather than silent.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            used = gpu_memory_used_mb()
            if used is None or max(used) < settle_mb:
                return
            time.sleep(2)

    def trim_log(self, keep_lines: int) -> None:
        """Keep the startup header, drop the per-request firehose.

        Everything diagnostic — the resolved engine args, the quantization method actually
        chosen, the KV-cache size, the chat template — is in the first few hundred lines. The
        rest is one line per request. Trim only after the process has exited, so nothing is
        appending.
        """
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
            if len(lines) <= keep_lines:
                return
            dropped = len(lines) - keep_lines
            self.log_path.write_text(
                "\n".join(lines[:keep_lines])
                + f"\n\n[harness] trimmed {dropped} lines of per-request logging; "
                  f"the startup header above is what carries the KV-cache/context evidence.\n"
            )
        except OSError:
            pass


def slot_context(run: dict[str, Any]) -> int:
    """The context ONE request actually gets.

    UNDER vLLM THIS IS JUST ``num_ctx``, AND THE DIVISION IS GONE. It used to return
    ``num_ctx // parallel``, because llama.cpp cut a fixed pre-allocated window into
    ``--parallel`` equal slots and a request only ever saw its own share (MEASURED: ``-c 16384
    --parallel 2`` reported ``n_ctx 8192`` per slot). Every caller that asks "does my prompt
    plus my token budget fit" was therefore asking about a number that SHRANK as concurrency
    rose, which is why raising ``parallel`` used to be a way to silently truncate every plan
    request at once.

    PagedAttention removes the tradeoff rather than moving it: KV is allocated per block as a
    sequence grows, so concurrency is bounded by the size of the KV pool (see
    ``ServerHandle.kv_cache``) and not by the per-request window. The function is KEPT, rather
    than inlined at its call sites, because "the context one request gets" is still a real
    concept that the runner and the report both need to name — it simply has a different
    answer now.

    UNDER llama.cpp THE DIVISION IS BACK, because the engine is back. `_build_argv_llamacpp`
    asks for `-c num_ctx * parallel` and llama.cpp cuts that single pre-allocated pool between
    the `--parallel` slots, so one request still sees only `num_ctx`. Returning the divided
    figure is what lets `check_slot_context` fail a run BEFORE it truncates every plan request
    — the failure mode this function was written for in the first place.

    SO THE ANSWER IS `num_ctx` UNDER BOTH ENGINES, but for opposite reasons, and the
    distinction matters when a launch fails. vLLM pages KV, so the window is free of
    concurrency. llama.cpp does not, so `_build_argv_llamacpp` BUYS the same guarantee by
    asking for `num_ctx * parallel` up front — and that allocation is what can refuse to fit.
    A llama.cpp run therefore fails at LOAD (`CUDA error: out of memory`) rather than at
    request time with a truncated window, which is the trade this harness wants: the old
    `num_ctx // parallel` behaviour silently shrank every request instead.
    """
    return int(run["num_ctx"])


# Chars per token for THIS harness's prompts, measured rather than assumed. The usual "4" is an
# English-prose figure and it badly underestimates the Mode B prompt, which is now a database
# export: braces, quotes, colons and course codes all tokenise far worse than prose. MEASURED
# 2026-07-30 against a live gemma4-e4b — a 33.7k-character Mode B prompt reported
# prompt_eval_count 11,458, i.e. 2.94 chars/token, against the 8,400 that /4 predicted. A 36%
# underestimate is not a rounding error here: it is the difference between `check` passing and a
# run silently truncating every plan request against the context.
#
# CARRIED OVER UNCHANGED, and it is worth saying why that is legitimate: the figure is a
# property of the PROMPTS and the tokenizer family, not of the inference engine. Both models
# kept their tokenizers across the GGUF -> compressed-tensors requantization.
CHARS_PER_TOKEN = 2.9


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def check_slot_context(run: dict[str, Any], longest_prompt_tokens: int) -> str | None:
    """Complain if the context cannot hold the longest prompt plus the biggest token budget.

    Returns a message, or None when it fits. This is now purely a ``num_ctx`` question:
    concurrency no longer shrinks it (see ``slot_context``), so the only way to fail is to
    genuinely ask for more window than was configured.
    """
    slot = slot_context(run)
    budget = max(int(run.get("max_plan_tokens", 0)), int(run.get("max_output_tokens", 0)))
    needed = longest_prompt_tokens + budget
    if needed <= slot:
        return None
    return (
        f"context is {slot} tokens (num_ctx {run['num_ctx']}), but the longest prompt "
        f"(~{longest_prompt_tokens}) plus the largest token budget ({budget}) needs ~{needed}. "
        f"Requests would be cut off by the CONTEXT instead of the budget — which is not a "
        f"model fault and is scored as if it were. Raise `num_ctx`, or lower the token "
        f"budgets. Note that raising `num_ctx` costs KV-cache headroom and therefore "
        f"concurrency; `parallel` no longer costs context at all."
    )


def engine(cfg: dict[str, Any]) -> str:
    """Which inference server this run launches: ``"llamacpp"`` (default) or ``"vllm"``.

    MODEL_EVAL_ENGINE WINS OVER config.yaml, the same precedence `--spec`/MODEL_EVAL_SPEC has
    and for the same reason: `run.py --engine` has to reach every server launch in the process
    and they do not share a call chain.
    """
    env = os.environ.get("MODEL_EVAL_ENGINE")
    name = (env or cfg.get("engine") or "llamacpp").strip().lower()
    if name not in ("llamacpp", "vllm"):
        raise SystemExit(f"unknown engine {name!r} — expected 'llamacpp' or 'vllm'")
    return name


def effective_parallel(cfg: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> int:
    """How many requests this run actually keeps in flight — `run.parallel`, unless the active
    engine caps it.

    WHY AN ENGINE MAY CAP IT. `run.parallel` is a statement about the WORKLOAD ("how many
    students are on the site at once"), and under vLLM it costs only KV out of a pool sized at
    runtime. Under llama.cpp it is a LOAD-TIME ALLOCATION of `num_ctx * parallel` tokens, so on
    a card this full the workload figure can simply be unbuyable: MEASURED 2026-08-26,
    qwen3.8-27b UD-Q6_K_M loads at parallel 3 (23,388 MiB) and does not load at 4.

    CAPPING IS NOT SILENT AND DOES NOT POLLUTE QUALITY. `runner` records the resolved number as
    `simulated_users` on every env record, which is the field that already exists to say which
    concurrency regime a file was written in — the same convention that keeps `parallel: 1` and
    `parallel: 4` vLLM runs apart. Latency columns do not pool across the setting; the sampler,
    prompts and schemas are identical, so quality does.
    """
    want = max(1, int(cfg["run"].get("parallel", 1)))
    for cap in (backend_cfg(cfg).get("max_parallel"),
                (model_cfg or {}).get("max_parallel")):
        # A MODEL MAY BE TIGHTER THAN ITS ENGINE. qwen3.8-flash-next holds ~56 GB of weights in
        # page cache and cannot use quantized KV, so every extra slot costs f16 KV against RAM
        # that is already gone — the engine-wide cap of 3 is unaffordable for that one entry.
        if cap and want > int(cap):
            want = int(cap)
    return want


def backend_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """The `llamacpp:` or `vllm:` block, whichever this run is using."""
    return cfg[engine(cfg)]


def model_dir(cfg: dict[str, Any], model_cfg: dict[str, Any]) -> Path:
    """Where this model's weights live on disk — a DIRECTORY under vLLM, a FILE under
    llama.cpp.

    The two engines do not load the same artifact and that is not cosmetic: vLLM wants a
    Hugging Face checkpoint directory (config.json plus safetensors shards, W4A16
    compressed-tensors here), llama.cpp wants one quantized `.gguf`. A model entry therefore
    carries both `model_path` and `gguf`, and which one is real depends on `engine`.
    """
    root = Path(backend_cfg(cfg)["models_root"])
    if engine(cfg) == "llamacpp":
        rel = model_cfg.get("model_path")
        if not rel:
            raise SystemExit(
                f"model {model_cfg['name']!r} has no `model_path:` in config.yaml — under "
                f"--engine llamacpp that is the .gguf to load, relative to models_root."
            )
        return root / rel
    # A MODEL MAY BE llama.cpp-ONLY. `qwen3.8-flash-next` has no vLLM-loadable checkpoint at
    # any size, so it carries no `model_path_vllm:` — say that, rather than raising KeyError
    # three frames deep in a launch.
    rel = model_cfg.get("model_path_vllm")
    if not rel:
        raise SystemExit(
            f"model {model_cfg['name']!r} has no `model_path_vllm:` in config.yaml, so it "
            f"cannot run under --engine vllm — it is a GGUF-only entry. Run it with "
            f"--engine llamacpp, or exclude it with --models / --brackets."
        )
    return root / rel


def build_argv(cfg: dict[str, Any], model_cfg: dict[str, Any], host: str) -> list[str]:
    """The single place a server command line is constructed, for whichever engine is active."""
    if engine(cfg) == "llamacpp":
        return _build_argv_llamacpp(cfg, model_cfg, host)
    return _build_argv_vllm(cfg, model_cfg, host)


def _spec_argv_llamacpp(cfg: dict[str, Any], model_cfg: dict[str, Any]) -> list[str]:
    """Speculative-decoding flags for `llama-server`, or `--spec-type none`.

    RESTORED 2026-08-26 from the pre-vLLM harness, where it was measured to pay:
    qwen3.6-27b went 18.65 -> 25.04 tok/s (+34%) at 96.7% draft acceptance on the ~1841-token
    plan JSON this harness actually produces. That gain scales with how expensive the TARGET is
    per token, which is why it was worth more on the 27B than on the 9B (+2.7%) — and why it is
    worth most of all on qwen3.8-flash-next, whose decode is bound by reading expert weights
    over DDR4 at ~22 tok/s.

    TWO FAMILIES, AND THE CHEAP ONE FITS THIS WORKLOAD BEST:

      draft-simple  a second small model proposes tokens. Costs VRAM for the draft's weights
                    and requires an IDENTICAL VOCABULARY — llama.cpp aborts at load otherwise,
                    which is why `draft:` is opt-in per model rather than global.
      ngram-*       proposes continuations copied from the PROMPT. Costs no weights, no VRAM
                    and no second model. It wins exactly when the output echoes the input, and
                    this harness's plan stages emit course codes that appear verbatim in the
                    rendered requirement database — the case it is built for.

    ⚠ QUALITY NUMBERS DO NOT POOL ACROSS THIS SETTING. Accepting a drafted token requires the
    target's sampler to agree with the proposal, so at temperature 0.15 a speculative run is a
    near — not exact — reproduction of the unspeculated path. The old harness CONFIRMED this,
    not theorised it: two 27B runs with identical prompt and seed returned different plans.
    `MODEL_EVAL_SPEC` (i.e. `run.py run --spec ...`) overrides the config per invocation.
    """
    spec = (cfg.get("llamacpp") or {}).get("speculative") or {}
    method = os.environ.get("MODEL_EVAL_SPEC") or model_cfg.get("spec_method") \
        or spec.get("method") or "none"
    # `--spec` on the CLI speaks the vLLM vocabulary (none/mtp/ngram/draft); map it onto
    # llama.cpp's richer `--spec-type` so one flag drives both engines.
    method = {"none": "none", "mtp": "draft-mtp", "ngram": "ngram-cache",
              "draft": "draft-simple"}.get(method, method)

    if method == "none":
        return ["--spec-type", "none"]

    # PER-MODEL OVERRIDES, because the right draft depth is a property of the TARGET's cost per
    # token, not of the proposer. MEASURED 2026-08-26 on qwen3.8-flash-next (IQ3_XXS, n_cpu_moe
    # 32, 12,255-token prompt): n_max 4 -> 23.7 tok/s, 8 -> 26.4, 16 -> 30.5, with acceptance
    # never dropping below 99.5%. A target whose every token costs a round-trip to system RAM
    # amortises a deep draft that a GPU-resident model would not.
    def _p(key, default):
        v = model_cfg.get(f"spec_{key}")
        return spec.get(key, default) if v is None else v

    argv = ["--spec-type", method,
            "--spec-draft-n-max", str(_p("n_max", 4)),
            "--spec-draft-n-min", str(_p("n_min", 0)),
            "--spec-draft-p-min", str(_p("p_min", 0.75))]

    if method.startswith("ngram"):
        # No draft model, no VRAM, nothing else to configure.
        return argv

    # A DRAFT MODEL IS A HARD DEPENDENCY, and a missing one is a load-time abort rather than a
    # graceful fallback — say so here instead of letting the server die with the draft's path
    # in a C++ error.
    draft = model_cfg.get("draft_gguf") or spec.get("draft_gguf")
    if not model_cfg.get("draft") or not draft:
        return ["--spec-type", "none"]
    draft_path = Path(cfg["llamacpp"]["models_root"]) / draft
    if not draft_path.is_file():
        raise SystemExit(
            f"model {model_cfg['name']!r} asks for {method} speculation but its draft gguf "
            f"{draft_path} does not exist. Download it, set `draft: false` on the entry, or "
            f"run with --spec none."
        )
    argv += [
        "-md", str(draft_path),
        # The draft's own KV. f16 by default: its whole job is to be small and fast, and a 0.8B
        # draft's cache is negligible either way.
        "-ctkd", str(spec.get("cache_type_k", "f16")),
        "-ctvd", str(spec.get("cache_type_v", "f16")),
    ]
    # ALL THE DRAFT'S LAYERS ON THE CARD. A partially-offloaded draft would add CPU-speed
    # latency to EVERY verification pass, which is the one thing speculation cannot afford.
    ngld = spec.get("n_gpu_layers_draft")
    if ngld is not None:
        argv += ["-ngld", str(ngld)]
    if spec.get("device_draft"):
        argv += ["-devd", str(spec["device_draft"])]
    argv += [str(a) for a in spec.get("extra_args", [])]
    return argv


def _build_argv_llamacpp(cfg: dict[str, Any], model_cfg: dict[str, Any],
                         host: str) -> list[str]:
    """`llama-server` command line.

    THE ONE THING THAT DIFFERS IN KIND FROM vLLM IS `-c`. llama.cpp pre-allocates a single KV
    pool of `-c` tokens and the `--parallel` slots SHARE it (the server logs
    `kv_unified = 'true'`), so a run that wants `parallel` concurrent requests each seeing
    `num_ctx` has to ask for `num_ctx * parallel`. MEASURED 2026-08-26: at `-c 16384
    --parallel 4`, four ~8.7k-token prompts died with `decode: Context size has been exceeded`;
    at `-c 65536 --parallel 4` the model would not load at all (`CUDA error: out of memory`,
    21.49 GiB of weights plus that pool against 23.55 GiB). Both failures are loud, which is
    the only mercy here — `slot_context` divides so `check_slot_context` predicts them first.
    """
    lc = cfg["llamacpp"]
    run = cfg["run"]
    parallel = effective_parallel(cfg, model_cfg)

    argv = [
        lc["server_exe"],
        "-m", str(model_dir(cfg, model_cfg)),
        # The name the client sees on /v1/models, pinned to the harness's short name so records
        # say `qwen3.8-27b` and not a path — same reason as vLLM's --served-model-name.
        "--alias", model_cfg["name"],
        "--host", host,
        "--port", str(lc["port"]),
        # THE WHOLE SHARED POOL, not the per-request window. See the docstring.
        "-c", str(int(run["num_ctx"]) * parallel),
        "--parallel", str(parallel),
        "-ngl", str(model_cfg.get("n_gpu_layers", lc.get("n_gpu_layers", 99))),
        # FLASH ATTENTION IS REQUIRED TO QUANTIZE THE V CACHE, not merely a speed knob — this
        # is why the pre-vLLM harness passed it unconditionally. Per-model because an
        # architecture whose attention llama.cpp implements specially may not support it;
        # qwen3.8-flash-next's hybrid Gated-DeltaNet/sparse path is exactly that shape.
        "-fa", str(model_cfg.get("flash_attn") or lc.get("flash_attn", "on")),
        "-ctk", str(model_cfg.get("cache_type_k") or lc.get("cache_type_k", "q8_0")),
        "-ctv", str(model_cfg.get("cache_type_v") or lc.get("cache_type_v", "q8_0")),
        "--seed", str(run["seed"]),
        # HOW THE WEIGHTS GET INTO MEMORY. `--mlock`/`--no-mmap` are deprecated in this build in
        # favour of one `--load-mode`. THE CHOICE IS PER MODEL AND IT MATTERS BOTH WAYS:
        #   mmap+mlock  pins weights in RAM — right for a model that FITS, wrong for one that
        #               does not, where it OOMs the box instead of paging.
        #   auto/mmap   lets the OS evict — the only survivable mode for qwen3.8-flash-next,
        #               whose 72.5-82 GB sits against ~61 GB of RAM plus the card.
        "-lm", str(model_cfg.get("load_mode") or lc.get("load_mode", "auto")),
        # Debug verbosity is REQUIRED, not optional: it is the only level at which this build
        # logs per-layer device assignment, and without that the harness cannot tell a fully
        # offloaded model from one silently running half on the CPU. Costs a few MB of log.
        "-lv", str(lc.get("log_verbosity", 5)),
        # The web UI is dead weight for a harness that only speaks HTTP, and its assets are
        # fetched at BUILD time — a build without them logs an error and serves a broken page.
        "--no-webui",
    ]

    # BATCH SIZES. Prefill is submitted `-b` tokens at a time and computed `-ub` at a time, and
    # on a model whose experts live in system RAM the physical batch decides how much work each
    # round of expert reads amortises. Left at llama.cpp's defaults (2048/512) unless a model
    # says otherwise, because raising `-ub` also raises the compute buffer — the allocation that
    # OOMs at long prompts.
    for flag, key in (("-b", "batch_size"), ("-ub", "ubatch_size"), ("-t", "threads")):
        val = model_cfg.get(key, lc.get(key))
        if val:
            argv += [flag, str(val)]

    # EXPERTS INTO SYSTEM RAM. The llama.cpp equivalent of vLLM's `--cpu-offload-gb`, but far
    # better targeted: it moves only the MoE expert tensors, which for an A4B/A6B model are the
    # bulk of the weights and the part that is READ SPARSELY (a handful of experts per token
    # rather than every byte every step). This is the flag that makes a checkpoint larger than
    # VRAM viable at all, and the reason a GGUF side-car exists next to vLLM.
    n_cpu_moe = model_cfg.get("n_cpu_moe")
    if n_cpu_moe:
        argv += ["-ncmoe", str(n_cpu_moe)]

    # PER-TENSOR PLACEMENT, the escape hatch `-ncmoe` is a shorthand for. `-ncmoe N` moves the
    # first N layers' experts wholesale; a regex here can put, say, only the `ffn_down_exps` of
    # every layer on the CPU and keep the rest resident. Unset by default because it is easy to
    # write one that silently matches nothing.
    for ot in model_cfg.get("override_tensor", []) or []:
        argv += ["-ot", str(ot)]

    # REASONING. llama.cpp keeps the launch-time switch vLLM dropped, so unlike the vLLM path
    # this genuinely turns the channel off rather than only deciding how it is parsed.
    budget = int(run.get("reasoning_budget", 0) or 0)
    if budget > 0:
        argv += ["--reasoning-budget", str(budget)]
    elif model_cfg.get("think") is False:
        argv += ["-rea", "off", "--reasoning-budget", "0"]

    argv += _spec_argv_llamacpp(cfg, model_cfg)
    argv += [str(a) for a in model_cfg.get("extra_args_llamacpp", [])]
    return argv


def _build_argv_vllm(cfg: dict[str, Any], model_cfg: dict[str, Any], host: str) -> list[str]:
    """The single place a vLLM command line is constructed.

    Everything that could differ between models and pollute the comparison is read from
    ``config.yaml`` and applied identically; only ``--model`` and the per-model overrides
    explicitly declared in the model's own entry vary.
    """
    vllm = cfg["vllm"]
    run = cfg["run"]

    argv = [
        vllm["server_exe"], "serve", str(model_dir(cfg, model_cfg)),
        # The name the client sees on /v1/models and sends as `model`. Pinned to the harness's
        # own short name so records, logs and config all say `qwen3.8-27b` rather than an
        # absolute path that changes if the checkpoint moves.
        "--served-model-name", model_cfg["name"],
        "--host", host,
        "--port", str(vllm["port"]),
        # THE PER-REQUEST WINDOW, and under vLLM every concurrent request gets all of it.
        # Both checkpoints declare max_position_embeddings 262144; pinning it here is what
        # makes the models comparable to each other and keeps the KV pool from being sized
        # against a window nothing in this harness ever uses.
        "--max-model-len", str(run["num_ctx"]),
        # HOW MANY STUDENTS MAY BE IN FLIGHT AT ONCE. Under llama.cpp this was `--parallel`
        # and it stole context from every request; here it is a scheduler bound and costs
        # nothing but KV, which is the point of the migration. `runner.run_model` keeps
        # exactly `run.parallel` requests in flight, so this stays in step with it.
        "--max-num-seqs", str(effective_parallel(cfg, model_cfg)),
        # HOW MUCH OF THE CARD THE ENGINE MAY CLAIM, weights and KV together. Not a tuning
        # knob to raise casually: vLLM profiles a forward pass to size the KV pool, and
        # anything left to the rest of the box has to cover the CUDA context and fragmentation.
        "--gpu-memory-utilization", str(vllm.get("gpu_memory_utilization", 0.92)),
        # THE DIRECT DESCENDANT of `--cache-type-k/v q8_0`. Halves KV bytes per token, which on
        # a 24 GB card holding a ~19.5 GB checkpoint is the difference between two concurrent
        # full-context sequences and five. Ampere has no native FP8 arithmetic, so this is a
        # STORAGE format that is dequantized into the kernel — the saving is memory, not math.
        #
        # PER-MODEL, BECAUSE IT IS NOT A PROPERTY OF THE CARD ALONE. It depends on which
        # attention backend the model's shape selects, and that varies per architecture:
        #
        #   ValueError: FP8 KV cache is not supported by the Triton attention backend on
        #   NVIDIA GeForce RTX 3090 Ti (compute capability 8.6)
        #
        # MEASURED 2026-08-22: qwen3.8-27b lands on FlashInfer and takes fp8 happily;
        # gemma4-26b lands on Triton attention and refuses it outright at engine start. This
        # is one of the few settings that legitimately differs between models here — the
        # comparison it could pollute is throughput, and a model that will not start pollutes
        # it worse.
        "--kv-cache-dtype",
        str(model_cfg.get("kv_cache_dtype") or vllm.get("kv_cache_dtype", "fp8")),
        # TEXT-ONLY, DECLARED. Both checkpoints are multimodal (Qwen3_5ForConditionalGeneration,
        # Gemma4ForConditionalGeneration) and this harness never sends an image or an audio
        # clip. Left at its default, vLLM sizes and reserves multimodal preprocessor/encoder
        # memory for a modality that will never arrive, and takes it straight out of the KV
        # pool. Zeroing the limits does NOT unload the vision tower — that is in the weights on
        # disk — it just stops paying for the path at runtime.
        "--limit-mm-per-prompt", '{"image":0,"video":0,"audio":0}',
        # Reproducibility, matching the sampler seed the client sends per request.
        "--seed", str(run["seed"]),
        # PREFIX CACHING, stated rather than left to the default. Every request here opens with
        # the same system prompt, and within a program with the same rendered requirement
        # database — thousands of tokens that would otherwise be re-prefilled per call. This is
        # the second reason the switch pays, after paged KV, and it is worth being explicit
        # about because it is also the reason TTFT here is not comparable to the llama.cpp-era
        # numbers: a cache hit skips work the old stack always did.
        "--enable-prefix-caching",
    ]

    # CPU WEIGHT OFFLOAD, per model. Streams the overflow of a checkpoint that does not fit in
    # VRAM from system RAM, over PCIe, on every forward pass. It is a LAST RESORT and the only
    # entry that uses it says why: qwen3.6-35b-a3b's usable quants are 25 GB against a 22.1 GB
    # budget, and the one small enough to fit (20.5 GB) left too little for a KV cache and did
    # not load anyway.
    #
    # IT COSTS BANDWIDTH, WHICH IS THE THING DECODE IS ALREADY BOUND BY, so a model using this
    # is not latency-comparable to one that does not — expect it to be the slowest row in the
    # table by a wide margin, and read its latency columns as a property of the offload rather
    # than of the model. An A3B MoE is the best case for it (~3B of 35B parameters active per
    # token, so most of what sits in system RAM is not read on any given step), which is the
    # only reason it is worth trying at all.
    offload = model_cfg.get("cpu_offload_gb")
    if offload:
        argv += ["--cpu-offload-gb", str(offload)]

    # REASONING. Under llama.cpp this was `--reasoning off --reasoning-budget 0` at launch,
    # because that was the only switch that behaved the same across every chat template. vLLM
    # has no launch-time off switch — a template that wants to reason will reason — so the
    # control moved entirely to the per-request `chat_template_kwargs: {enable_thinking: false}`
    # the client already sends, and this flag only decides whether the output arrives SPLIT
    # into a `reasoning_content` channel or inline in `content` as raw <think> tags.
    #
    # Naming a parser is therefore not "turning thinking on": it is making sure that if a model
    # emits reasoning anyway, it lands in the channel `GenerationResult.reasoning_text` reads
    # and the scorers do not have to strip tags out of the answer. The `plan_b_thinking` arm
    # that used to be the other caller of this was removed 2026-08-24; `run.advising_thinking`
    # is the only switch left that asks any stage to reason.
    parser = model_cfg.get("reasoning_parser") or vllm.get("reasoning_parser")
    if parser:
        argv += ["--reasoning-parser", str(parser)]


    # TOOLS. Only the QA/explain stages offer any (see the client's tools section); a model
    # with no parser configured simply never emits a parseable call and the stage answers
    # without one, which is the same degradation the old stack had.
    tool_parser = model_cfg.get("tool_call_parser") or vllm.get("tool_call_parser")
    if tool_parser:
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", str(tool_parser)]

    # SPECULATIVE DECODING. Off unless a model declares it (or MODEL_EVAL_SPEC overrides), and
    # it is per-model because only one of these two models has anything to speculate WITH.
    #
    # THE llama.cpp ERA'S VERSION OF THIS IS GONE, not ported: that used a separate 0.8B draft
    # gguf, and it paid by amortizing weight reads on hardware where the target model was split
    # across two power-limited cards. The equivalents here are different in kind:
    #
    #   "mtp"   — the multi-token-prediction head shipped INSIDE the qwen3.8-27b checkpoint
    #             (`model-mtp.safetensors`). vLLM resolves the draft from the target model
    #             itself, so no second checkpoint is configured. Costs ~0.85 GB of weights,
    #             which comes straight out of the KV pool and therefore out of concurrency.
    #   "ngram" — prompt-lookup decoding. Proposes continuations copied from the prompt, so it
    #             costs NO VRAM at all and wins exactly when the output echoes the input. This
    #             harness's plan stages emit course codes that appear verbatim in the rendered
    #             requirement database, which is the case it is built for.
    #
    # ⚠ SPECULATION AND BATCHING COMPETE FOR THE SAME SPARE COMPUTE. A drafted token is only
    # cheap while the GPU has idle capacity during decode, and `run.parallel > 1` is already
    # spending that capacity on other students. Measure this at LOW concurrency or the result
    # will say "speculation does not help" when what it means is "batching got there first".
    spec = _speculative_config(cfg, model_cfg)
    if spec:
        argv += ["--speculative-config", json.dumps(spec)]

    argv += [str(a) for a in model_cfg.get("extra_args", [])]
    return argv


def _has_mtp_head(cfg: dict[str, Any], model_cfg: dict[str, Any]) -> bool:
    """Does this checkpoint actually carry a multi-token-prediction head?

    CHECKED ON DISK rather than assumed from the model name. A model without one fails at
    engine start, which through a sweep is an expensive way to learn a static fact — the
    config.json key and the weight file are both readable in microseconds.
    """
    path = model_dir(cfg, model_cfg)
    if list(path.glob("*mtp*.safetensors")):
        return True
    try:
        cfg_json = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False
    text = cfg_json.get("text_config", cfg_json)
    return any("mtp" in k.lower() or "nextn" in k.lower() for k in text)


def _speculative_config(cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The `--speculative-config` payload for this model, or None for plain decoding.

    `MODEL_EVAL_SPEC` wins over config.yaml so one sweep can be re-run under a different
    proposer without editing the file — the same override shape `--major` uses, and the reason
    it is an env var is that it has to reach every server launch in the process (the sweep's,
    and the thinking experiment's) and they do not share a call chain. `none` forces it off.
    """
    override = os.environ.get("MODEL_EVAL_SPEC")
    method = override if override else (model_cfg.get("speculative") or {}).get("method")
    if not method or method == "none":
        return None
    spec = dict((model_cfg.get("speculative") or {}))
    spec.pop("method", None)
    defaults = {"ngram": {"num_speculative_tokens": 4, "prompt_lookup_max": 8,
                          "prompt_lookup_min": 2},
                "mtp": {"num_speculative_tokens": 2},
                "draft_model": {"num_speculative_tokens": 4}}
    if method == "mtp" and not _has_mtp_head(cfg, model_cfg):
        # DEGRADE, DO NOT RAISE — same reasoning as the missing-draft branch below. Only
        # qwen3.8-27b ships a multi-token-prediction head; asking gemma4-26b for `mtp` would
        # die at engine start and, because `_run_sweep` catches per PROGRAM rather than per
        # model, take the other model's records down with it.
        print(f"[spec] {model_cfg['name']} has no MTP head in its checkpoint — running it "
              f"with plain decoding for this arm")
        return None

    if method in ("draft", "draft_model"):
        # A SEPARATE SMALL MODEL PROPOSES, the big one verifies — the direct descendant of the
        # llama.cpp `-md` draft gguf, and the only speculative method here that needs its own
        # checkpoint on disk.
        #
        # THE DRAFT MUST SHARE THE TARGET'S TOKENIZER, not merely its vocabulary SIZE. It
        # proposes token IDs that the target then scores, so a draft trained on a different
        # tokenization proposes ids that mean something else and every one gets rejected —
        # which costs time and returns nothing. `run.py check` verifies this by encoding the
        # same string with both tokenizers and comparing the ids, rather than trusting that two
        # models from the same family agree.
        method = "draft_model"
        draft = model_cfg.get("draft_model")
        if not draft:
            # DEGRADE, DO NOT RAISE. `--spec draft` is a request about HOW to decode, and only
            # some models have a tokenizer-compatible small sibling to decode with. Raising
            # here failed the whole PROGRAM — `_run_sweep` catches per program, not per model —
            # so one model lacking a draft cost the other model its results too. MEASURED
            # 2026-08-22: a `--spec draft` sweep produced 0-1 records per program across four
            # programs, because gemma4-26b aborted each one before qwen3.8-27b's records were
            # written.
            #
            # Falling back to plain decoding keeps that model in the run and measurable. The
            # env record's `speculative` field says which models actually speculated, so a
            # mixed run is still readable — and a silently-unspeculated model is exactly what
            # that field exists to make visible.
            print(f"[spec] {model_cfg['name']} has no `draft_model:` — running it with plain "
                  f"decoding for this arm (only models with a tokenizer-compatible small "
                  f"sibling can draft)")
            return None
        spec["model"] = str(Path(cfg["vllm"]["models_root"]) / draft)
    out = {**defaults.get(method, {}), **spec, "method": method}
    return out


def _server_env(cfg: dict[str, Any]) -> dict[str, str]:
    """The environment vLLM is launched into: this process's, plus the venv's `bin/` on PATH.

    THE VENV IS NOT ACTIVATED, deliberately — the harness invokes `vllm` by absolute path so it
    does not care which interpreter is running `run.py`. But vLLM SHELLS OUT to build tools
    during engine startup, and those are console scripts installed alongside `vllm` in the same
    `bin/`. Without this, the launch dies minutes in with

        FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
        RuntimeError: Engine core initialization failed.

    which is a PATH problem wearing the costume of a broken model: it happens after the weights
    load and the compile begins, and the only clue is one line buried in ~400 of traceback.
    MEASURED here 2026-08-22 on the very first vLLM launch attempt.

    Prepended rather than appended, so the venv's tools win over any system copy — the same
    precedence activating the venv would give.
    """
    env = dict(os.environ)
    exe = Path(backend_cfg(cfg)["server_exe"]).resolve()
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")

    # llama.cpp NEEDS THE CUDA LIBRARIES ON LD_LIBRARY_PATH, and they are not where a linker
    # would look. There is no system CUDA on this box and no sudo to install one, so the build
    # links against a toolkit assembled out of pip wheels (see `.llamacpp-env.sh`, which the
    # build itself sources). `llama-server` therefore starts with
    #
    #     error while loading shared libraries: libcudart.so.13
    #
    # unless the same directory is exported here. Same class of failure as the `ninja` one
    # above: it looks like a broken binary and it is a path problem.
    if engine(cfg) == "llamacpp":
        libdir = cfg["llamacpp"].get("cuda_lib_dir") or str(
            Path("/home/wylin/ai-academic-advisor/.cudatoolkit/lib64"))
        if Path(libdir).is_dir():
            env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def start_server(
    cfg: dict[str, Any], model_cfg: dict[str, Any], log_dir: Path, *, host: str | None = None,
) -> ServerHandle:
    eng = engine(cfg)
    backend = backend_cfg(cfg)
    host = host or resolve_host(backend.get("host", "auto"))
    argv = build_argv(cfg, model_cfg, host)
    base_url = f"http://{host}:{backend['port']}"

    # WHAT EACH ENGINE LOADS IS A DIFFERENT KIND OF THING ON DISK — a checkpoint directory for
    # vLLM, one .gguf file for llama.cpp — so the "missing weights" check is per-engine too.
    path = model_dir(cfg, model_cfg)
    if eng == "llamacpp":
        if not path.is_file():
            raise FileNotFoundError(
                f"model {model_cfg['name']} points at {path}, which is not a file. "
                f"llama.cpp loads a single quantized .gguf, not a checkpoint directory — "
                f"check the entry's `gguf:` path, or run with --engine vllm."
            )
    elif not path.is_dir():
        raise FileNotFoundError(
            f"model {model_cfg['name']} points at {path}, which is not a directory. "
            f"vLLM loads a Hugging Face checkpoint DIRECTORY (config.json + safetensors "
            f"shards), not a single quantized file the way llama.cpp loads a .gguf — "
            f"`run.py check` reports which entries are missing."
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{eng}_{model_cfg['name']}.log"
    log_file = log_path.open("w")
    process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT,
                               env=_server_env(cfg))
    handle = ServerHandle(process=process, log_path=log_path, argv=argv, base_url=base_url)

    client = VllmClient(base_url)
    # STARTUP IS SLOW AND THAT IS NORMAL, so the timeout is generous. Beyond loading ~19 GB of
    # weights, vLLM profiles a forward pass to size the KV pool and compiles/captures CUDA
    # graphs; the first launch of a given model also populates a compilation cache, so a cold
    # first start can be minutes longer than every start after it.
    deadline = time.time() + backend.get("startup_timeout_s", 1800)
    while time.time() < deadline:
        if process.poll() is not None:
            tail = "\n".join(handle.log_text().splitlines()[-30:])
            # THE ONE FAILURE WORTH TRANSLATING. llama.cpp pre-allocates `num_ctx * parallel`
            # tokens of KV at load, so raising `parallel` can make a model that ran fine stop
            # loading at all — and it reports that as a bare `CUDA error: out of memory` with
            # no hint that concurrency, not the model, is what got too big. MEASURED
            # 2026-08-26 on qwen3.8-27b UD-Q6_K_M (21.49 GiB) against 23.55 GiB:
            # parallel 3 loads at 23,388 MiB, parallel 4 does not load.
            if eng == "llamacpp" and "out of memory" in tail.lower():
                parallel = effective_parallel(cfg, model_cfg)
                num_ctx = int(cfg["run"]["num_ctx"])
                raise RuntimeError(
                    f"llama.cpp ran out of VRAM loading {model_cfg['name']}.\n\n"
                    f"It pre-allocates ONE KV pool of num_ctx * parallel = "
                    f"{num_ctx} * {parallel} = {num_ctx * parallel} tokens at load, on top of "
                    f"the weights — unlike vLLM, which pages KV and pays for concurrency out "
                    f"of a pool it sizes at runtime. Either:\n"
                    f"  - lower `run.parallel` (3 fits qwen3.8-27b UD-Q6_K_M on this card, "
                    f"4 does not), or\n"
                    f"  - use a smaller quant for this model's `gguf:`, or\n"
                    f"  - set this model's `n_gpu_layers` below {model_cfg.get('n_gpu_layers', 99)} "
                    f"to spill layers to RAM, or `n_cpu_moe` if it is an MoE, or\n"
                    f"  - run this model under --engine vllm.\n\n"
                    f"Last lines:\n{tail}"
                )
            raise RuntimeError(
                f"{eng} exited during startup for {model_cfg['name']} "
                f"(code {process.returncode}). Last lines:\n{tail}"
            )
        if client.health():
            return handle
        time.sleep(2)

    handle.stop()
    tail = "\n".join(handle.log_text().splitlines()[-30:])
    raise TimeoutError(
        f"{eng} for {model_cfg['name']} never became healthy at {base_url}. "
        f"Last lines:\n{tail}"
    )
