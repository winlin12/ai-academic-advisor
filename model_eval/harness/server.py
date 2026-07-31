"""llama-server process lifecycle.

WHY THE HARNESS OWNS THE PROCESS. Under Ollama, one long-lived daemon served every model and
``num_ctx`` travelled with each request. llama.cpp inverts that: one process serves exactly
one model, and context size, KV-cache type, GPU-layer count and reasoning mode are all fixed
at launch. If the harness merely *connected* to whatever server happened to be running, the
central promise — "every model sees identical settings" — would depend on whoever typed the
last command line. So the harness starts and stops the server itself, from ``config.yaml``,
and records the resulting command line in ``meta_*.json``.

TWO SUPPORTED TOPOLOGIES, picked from whether ``server_exe`` ends in ``.exe``:

1. NATIVE LINUX — a binary built from ./llama.cpp, running in this WSL box. Loopback works,
   paths are passed through unchanged, SIGTERM stops it. Nothing special happens.

2. WINDOWS BINARY OVER WSL INTEROP — the b10083 CUDA build on D:. This box has no CUDA
   toolkit in WSL, so this is the topology that actually runs today. Three things differ, and
   all three are silent failures if you get them wrong:

     - PATHS. The .exe cannot open ``/mnt/d/...``; every path on the command line is
       translated to ``D:\\...`` via ``wslpath -w``.
     - NETWORKING. WSL2 is NAT'd here (no ``networkingMode=mirrored`` in .wslconfig), so the
       server is NOT on loopback from this side. It binds 0.0.0.0 on the Windows side and the
       harness connects to the default-route gateway. Verified 2026-07-25: 127.0.0.1 refuses,
       172.30.192.1 answers, and Windows Firewall does not block it.
     - TERMINATION. SIGTERM kills only the Linux-side interop proxy — MEASURED, the .exe
       survived it still holding 4.7 GB of VRAM and the port. Stopping therefore goes through
       ``taskkill.exe``, or every model after the first would launch into an occupied port and
       a polluted VRAM baseline.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llamacpp_client import LlamaCppClient

# Offload truth from llama-server's own stderr. The Ollama harness had to shell out to
# journalctl and parse a summary line; here it is the server's own output, and this build
# (b10083) is more precise than the old summary: it names the device for EVERY layer, so
# "how much is on the card" is counted rather than inferred.
#
#   load_tensors: layer  27 assigned to device CUDA0, is_swa = 0
#
# Those lines are debug-level and need `-lv 5` (see log_verbosity in config.yaml) — at the
# default verbosity llama-server prints no offload information at all, which is precisely how
# a partially-offloaded model would otherwise sneak into the results unnoticed.
_LAYER_RE = re.compile(r"layer\s+(\d+) assigned to device (\w+)")
# Older/other builds print only a summary; kept so the parser doesn't depend on one build.
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_GPU_DEVICE_PREFIXES = ("CUDA", "ROCM", "HIP", "METAL", "VULKAN", "SYCL")
_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"  # WSL's CUDA shim; falls back to PATH


def gpu_memory_used_mb() -> list[int] | None:
    exe = _NVIDIA_SMI if Path(_NVIDIA_SMI).exists() else shutil.which("nvidia-smi")
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


# Enough of the GGUF header to answer one question: how many blocks (transformer layers) does
# this model have? Needed because llama.cpp counts MoE offload from the BOTTOM of the stack
# (see resolve_n_cpu_moe) and the harness wants to express it from the top. Hand-rolled rather
# than importing llama.cpp/gguf-py: that package is a submodule with its own numpy dependency,
# and this reads one integer out of the first few KB of the file.
#
# Layout: "GGUF" | u32 version | u64 tensor_count | u64 kv_count | kv_count x (key, type, value)
# where a key is u64 length + utf-8 bytes, and a value is typed by the enum below.
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def gguf_block_count(path: str | Path) -> int | None:
    """The model's layer count from its own gguf metadata, or None if unreadable.

    The key is architecture-prefixed (``gemma4.block_count``, ``qwen3moe.block_count``, ...),
    so it is matched by suffix. Returning None rather than raising keeps a malformed or
    truncated header from taking down a run that did not ask for layer-relative offload;
    the caller decides whether it actually needed the number.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

            def read(fmt: str) -> Any:
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            def skip_value(vtype: int) -> None:
                if vtype in _GGUF_SCALAR_FMT:
                    f.seek(struct.calcsize(_GGUF_SCALAR_FMT[vtype]), 1)
                elif vtype == _GGUF_STRING:
                    f.seek(read("<Q"), 1)
                elif vtype == _GGUF_ARRAY:
                    elem_type, count = read("<I"), read("<Q")
                    if elem_type == _GGUF_STRING:      # tokenizer vocab lives here — 150k+
                        for _ in range(count):         # entries, each length-prefixed
                            f.seek(read("<Q"), 1)
                    elif elem_type in _GGUF_SCALAR_FMT:
                        f.seek(struct.calcsize(_GGUF_SCALAR_FMT[elem_type]) * count, 1)
                    else:
                        raise ValueError(f"unsupported gguf array element type {elem_type}")
                else:
                    raise ValueError(f"unsupported gguf value type {vtype}")

            for _ in range(n_kv):
                key = f.read(read("<Q")).decode("utf-8", "replace")
                vtype = read("<I")
                if key.endswith(".block_count") and vtype in _GGUF_SCALAR_FMT:
                    return int(read(_GGUF_SCALAR_FMT[vtype]))
                skip_value(vtype)
    except (OSError, struct.error, ValueError):
        return None
    return None


def resolve_n_cpu_moe(
    run: dict[str, Any], model_cfg: dict[str, Any], gguf_path: str | Path
) -> int | None:
    """Turn whichever MoE-offload knob was configured into the ``--n-cpu-moe N`` llama.cpp wants.

    llama.cpp has exactly one MoE-offload count and it points the wrong way for comparing
    models: ``--n-cpu-moe N`` keeps the experts of the FIRST N layers in system RAM, so N is
    "how much is *not* on the card" and its meaning changes with the model's depth. 35 is a
    saturating no-op on a 30-block model and leaves half the experts resident on a 64-block
    one — the same config line describing two different machines.

    ``n_gpu_moe`` states the other end: keep the experts of N layers on the GPU. That is the
    quantity VRAM is actually spent on, so it is the one that transfers between models, and
    the harness resolves it per model as ``block_count - n_gpu_moe``.

    Per-model entries override the run-level default; the two knobs are mutually exclusive
    within a level, since they would otherwise silently disagree about the same split.
    """
    for source, where in ((model_cfg, f"model {model_cfg.get('name', '?')}"), (run, "run")):
        n_gpu_moe, n_cpu_moe = source.get("n_gpu_moe"), source.get("n_cpu_moe")
        if n_gpu_moe is not None and n_cpu_moe is not None:
            raise ValueError(
                f"{where} sets both n_gpu_moe ({n_gpu_moe}) and n_cpu_moe ({n_cpu_moe}); "
                "they describe the same split from opposite ends — set one."
            )
        if n_cpu_moe is not None:
            return int(n_cpu_moe)
        if n_gpu_moe is not None:
            blocks = gguf_block_count(gguf_path)
            if blocks is None:
                raise ValueError(
                    f"{where} sets n_gpu_moe={n_gpu_moe}, which needs the model's layer count, "
                    f"but no block_count could be read from {gguf_path}. Use n_cpu_moe instead."
                )
            # Clamped, not an error: n_gpu_moe >= depth is the legitimate way to say "all
            # experts on the card", and 0 is --cpu-moe.
            return max(0, blocks - int(n_gpu_moe))
    return None


def is_windows_exe(server_exe: str | Path | None) -> bool:
    """Which of the two topologies in the module docstring we are in."""
    return bool(server_exe) and str(server_exe).lower().endswith(".exe")


def to_server_path(path: str | Path, *, windows: bool) -> str:
    """Render a path the way the *server process* will have to open it.

    Under interop the .exe is a Windows program handed a Linux command line: ``/mnt/d/...``
    is not a path it can resolve. ``wslpath -w`` is the authority on the translation rather
    than a hand-rolled /mnt/<drive> rewrite, so unusual mounts keep working.
    """
    if not windows:
        return str(path)
    try:
        return subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not translate {path} to a Windows path: {exc}") from exc


def _default_gateway() -> str | None:
    """The Windows host's address on the NAT network, from the default route."""
    try:
        out = subprocess.run(
            ["ip", "route"], capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
            return parts[2]
    return None


def resolve_host(configured: str, server_exe: str | Path | None = None) -> str:
    """The address the *harness* connects to (not necessarily the one the server binds).

    Native Linux: ``auto`` -> loopback, anything else literal.

    Windows interop: ``auto`` -> the NAT gateway, because under NAT the Windows host is not on
    this box's loopback and dialling 127.0.0.1 fails as a connection refusal several minutes
    into a model load — which reads like a broken model rather than a broken address.

    AN EXPLICIT VALUE IS ALWAYS HONOURED, including 127.0.0.1. That is not an oversight:
    setup/allow_wsl_llamacpp.ps1 documents `networkingMode=mirrored` + `host: 127.0.0.1` as the
    supported firewall-free alternative, and under mirrored networking loopback is exactly
    right. Overriding it here would break that configuration for the sake of catching a typo.
    """
    if configured and configured != "auto":
        return configured
    if is_windows_exe(server_exe):
        gateway = _default_gateway()
        if not gateway:
            raise RuntimeError(
                "server_exe is a Windows binary but no default route was found, so the "
                "address of the Windows host is unknown. Set llamacpp.host explicitly."
            )
        return gateway
    return "127.0.0.1"


@dataclass
class ServerHandle:
    process: subprocess.Popen
    log_path: Path
    argv: list[str]
    base_url: str
    started_at: float = field(default_factory=time.time)
    # True when `process` is only the interop proxy for a Windows .exe; changes how stop() works.
    windows: bool = False

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def offload(self) -> dict[str, Any]:
        log = self.log_text()

        # Preferred: count the per-layer device assignments. Last assignment wins, since a
        # model can be re-laid-out during load.
        assignments: dict[int, str] = {}
        for index, device in _LAYER_RE.findall(log):
            assignments[int(index)] = device.upper()
        if assignments:
            total = len(assignments)
            on_gpu = sum(
                1 for d in assignments.values()
                if d.startswith(_GPU_DEVICE_PREFIXES)
            )
            cpu_layers = sorted(i for i, d in assignments.items()
                                if not d.startswith(_GPU_DEVICE_PREFIXES))
            return {
                "source": "llama_server_layer_assignment",
                "layers": f"{on_gpu}/{total}",
                "fully_offloaded": on_gpu == total,
                "cpu_layers": cpu_layers[:20],
            }

        matches = _OFFLOAD_RE.findall(log)
        if matches:
            on_gpu, total = map(int, matches[-1])
            return {
                "source": "llama_server_summary_line",
                "layers": f"{on_gpu}/{total}",
                "fully_offloaded": on_gpu == total,
            }

        # Not knowing is reported as not knowing. A model silently running half on the CPU
        # would post terrible latency and identical quality, and the report needs to be able
        # to say which of those it is looking at.
        return {"source": "unverified", "layers": None, "fully_offloaded": None}

    def stop(self, timeout: float = 30.0, *, trim_log_lines: int | None = None) -> None:
        # Under interop, terminate() reaches only the Linux-side proxy: MEASURED 2026-07-25,
        # the .exe outlived it holding 4.7 GB of VRAM and port 8099. Kill the Windows process
        # FIRST, so the proxy's own exit is the confirmation rather than a hopeful guess.
        if self.windows:
            self._taskkill()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        if trim_log_lines:
            self.trim_log(trim_log_lines)

    @staticmethod
    def _taskkill() -> None:
        """Kill llama-server.exe by image name.

        BY IMAGE NAME, NOT PID, and that is deliberate: llama-server does not report its
        Windows PID anywhere the harness can read, and a tasklist diff would race. The harness
        already claims exclusive ownership of the server lifecycle and of port 8099, so "every
        llama-server.exe on this box" and "the one I started" are the same set — but note that
        this WILL take down a llama-server.exe you started by hand in another window.
        """
        exe = "/mnt/c/Windows/System32/taskkill.exe"
        if not Path(exe).exists():
            exe = shutil.which("taskkill.exe") or "taskkill.exe"
        try:
            subprocess.run([exe, "/IM", "llama-server.exe", "/F"],
                           capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass  # nothing left to do; startup of the next model reports the port if it stuck

    def trim_log(self, keep_lines: int) -> None:
        """Keep the startup header, drop the per-request firehose.

        `-lv 5` is mandatory to get per-layer offload out of this build, but it also logs
        every slot operation of every request — ~90 MB per model, which is over a gigabyte for
        a full sweep and buys nothing once offload has been recorded. Everything diagnostic
        (device selection, layer assignment, context size, chat template) is in the first few
        hundred lines. Trim only after the process has exited, so nothing is appending.
        """
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
            if len(lines) <= keep_lines:
                return
            dropped = len(lines) - keep_lines
            self.log_path.write_text(
                "\n".join(lines[:keep_lines])
                + f"\n\n[harness] trimmed {dropped} lines of per-request logging; "
                  f"the startup header above is what carries the offload/context evidence.\n"
            )
        except OSError:
            pass


def slot_context(run: dict[str, Any]) -> int:
    """The context ONE request actually gets: ``num_ctx // parallel``.

    The number that matters for "does my prompt plus my token budget fit", and the one that is
    easy to get wrong, because config names `num_ctx` and llama.cpp reports the slot's share.
    """
    return int(run["num_ctx"]) // max(1, int(run.get("parallel", 1)))


# Chars per token for THIS harness's prompts, measured rather than assumed. The usual "4" is an
# English-prose figure and it badly underestimates the Mode B prompt, which is now a database
# export: braces, quotes, colons and course codes all tokenise far worse than prose. MEASURED
# 2026-07-30 against a live gemma4-e4b — a 33.7k-character Mode B prompt reported
# prompt_eval_count 11,458, i.e. 2.94 chars/token, against the 8,400 that /4 predicted. A 36%
# underestimate is not a rounding error here: it is the difference between `check` passing and a
# run silently truncating every plan request against the context.
CHARS_PER_TOKEN = 2.9


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def check_slot_context(run: dict[str, Any], longest_prompt_tokens: int) -> str | None:
    """Complain if a slot cannot hold the longest prompt plus the biggest token budget.

    Returns a message, or None when it fits. This is the failure `--parallel` invites: at
    num_ctx 16384 two slots is 8192 and still fits a ~2.9k prompt with a 4096 budget, but four
    slots is 4096 — smaller than the budget alone — and every plan request would be truncated by
    the context rather than by the budget, silently, on every model at once.
    """
    slot = slot_context(run)
    budget = max(int(run.get("max_plan_tokens", 0)), int(run.get("max_output_tokens", 0)))
    needed = longest_prompt_tokens + budget
    if needed <= slot:
        return None
    return (
        f"per-slot context is {slot} tokens (num_ctx {run['num_ctx']} / parallel "
        f"{run.get('parallel', 1)}), but the longest prompt (~{longest_prompt_tokens}) plus the "
        f"largest token budget ({budget}) needs ~{needed}. Requests would be cut off by the "
        f"CONTEXT instead of the budget — which is not a model fault and is scored as if it "
        f"were. Lower `parallel`, raise `num_ctx`, or lower the token budgets."
    )


def build_argv(cfg: dict[str, Any], model_cfg: dict[str, Any], host: str) -> list[str]:
    """The single place a llama-server command line is constructed.

    Everything that could differ between models and pollute the comparison is read from
    ``config.yaml`` and applied identically; only ``--model`` and the per-model overrides
    explicitly declared in the model's own entry vary.
    """
    llama = cfg["llamacpp"]
    run = cfg["run"]
    windows = is_windows_exe(llama["server_exe"])
    gguf_local = Path(llama["models_root"]) / model_cfg["gguf"]   # readable from this side
    gguf = to_server_path(gguf_local, windows=windows)            # openable by the server

    # The bind address is not the connect address under interop: the server binds on the
    # Windows side, where this box's loopback means the wrong machine. 0.0.0.0 is what makes
    # it reachable across the NAT boundary; `host` (the gateway) is what the client dials.
    bind_host = "0.0.0.0" if windows else host

    argv = [
        llama["server_exe"],
        "-m", gguf,

        "--host", bind_host,
        "--port", str(llama["port"]),
        "-c", str(run["num_ctx"]),       # fixed at launch under llama.cpp — the whole reason
        "--n-gpu-layers", str(run["n_gpu_layers"]),
        "--flash-attn", "on",            # required to quantize the V cache
        "--cache-type-k", run["cache_type_k"],
        "--cache-type-v", run["cache_type_v"],
        "--jinja",                       # each gguf's own chat template, exactly like the app
        # SLOTS. llama.cpp divides the context between them — the per-request window is
        # `--ctx-size / --parallel`, not `--ctx-size` (MEASURED: -c 16384 --parallel 2 reports
        # n_ctx 8192 per slot). So raising this silently shrinks what a single request may use,
        # and `slot_context()` below is what stops that going unnoticed.
        #
        # The runner issues one request at a time, so anything above 1 sits idle while costing
        # context. It is here so the eval can be run at the deployment's real slot count.
        "--parallel", str(run.get("parallel", 1)),
        "--no-mmap",
        "--no-webui",
        # Debug verbosity is REQUIRED, not optional: it is the only level at which this build
        # logs per-layer device assignment, and without that the harness cannot tell a fully
        # offloaded model from one silently running half on the CPU. Costs a few MB of log.
        "-lv", str(llama.get("log_verbosity", 5)),
    ]
    # MoE expert offload to CPU. A no-op on dense models, so it is applied uniformly rather
    # than conditionally — one fewer axis on which models differ. Configured either as
    # `n_cpu_moe` (llama.cpp's own bottom-up count) or `n_gpu_moe` (layers of experts kept on
    # the card, resolved against this model's depth); see resolve_n_cpu_moe.
    n_cpu_moe = resolve_n_cpu_moe(run, model_cfg, gguf_local)
    if n_cpu_moe is not None:
        argv += ["--n-cpu-moe", str(n_cpu_moe)]
    # Reasoning off at launch: the only switch that behaves the same across Qwen3, gpt-oss
    # and GLM templates. Per-request chat_template_kwargs is a backstop in the client.
    if model_cfg.get("think") is False:
        argv += ["--reasoning", "off", "--reasoning-budget", "0"]
    # --mlock OOMs on models larger than this box's RAM; see config's mlock_max_gb.
    if model_cfg.get("mlock", True):
        argv.append("--mlock")
    argv += [str(a) for a in model_cfg.get("extra_args", [])]
    return argv


def start_server(
    cfg: dict[str, Any], model_cfg: dict[str, Any], log_dir: Path, *, host: str | None = None
) -> ServerHandle:
    windows = is_windows_exe(cfg["llamacpp"]["server_exe"])
    host = host or resolve_host(cfg["llamacpp"].get("host", "auto"),
                                cfg["llamacpp"]["server_exe"])
    argv = build_argv(cfg, model_cfg, host)
    base_url = f"http://{host}:{cfg['llamacpp']['port']}"

    # A survivor from an earlier crashed run owns the port and skews the VRAM baseline that
    # the next model's delta is measured against. Cheap to rule out, expensive to debug.
    if windows:
        ServerHandle._taskkill()
        time.sleep(2)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-server_{model_cfg['name']}.log"
    log_file = log_path.open("w")
    process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    handle = ServerHandle(process=process, log_path=log_path, argv=argv, base_url=base_url,
                          windows=windows)

    client = LlamaCppClient(base_url)
    deadline = time.time() + cfg["llamacpp"].get("startup_timeout_s", 900)
    while time.time() < deadline:
        if process.poll() is not None:
            tail = "\n".join(handle.log_text().splitlines()[-25:])
            raise RuntimeError(
                f"llama-server exited during startup for {model_cfg['name']} "
                f"(code {process.returncode}). Last lines:\n{tail}"
            )
        if client.health():
            return handle
        time.sleep(2)

    handle.stop()
    tail = "\n".join(handle.log_text().splitlines()[-25:])
    raise TimeoutError(
        f"llama-server for {model_cfg['name']} never became healthy at {base_url}. "
        f"Last lines:\n{tail}"
    )
