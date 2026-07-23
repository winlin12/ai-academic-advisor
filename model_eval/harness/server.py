"""llama-server process lifecycle, driven from WSL over Windows interop.

WHY THE HARNESS OWNS THE PROCESS. Under Ollama, one long-lived daemon served every model and
``num_ctx`` travelled with each request. llama.cpp inverts that: one process serves exactly
one model, and context size, KV-cache type, GPU-layer count and reasoning mode are all fixed
at launch. If the harness merely *connected* to whatever server happened to be running, the
central promise — "every model sees identical settings" — would depend on whoever typed the
last command line. So the harness starts and stops the server itself, from ``config.yaml``,
and records the resulting command line in ``meta_*.json``.

WHY THE ARGV IS BUILT HERE AND NOT IN ``D:\\llm\\run-model.ps1``. That launcher is a fine
hand-driving convenience, but it lives outside the repo, so a run driven by it is not
reproducible from a checkout. ``llama-server.exe`` is directly executable from WSL (Windows
interop), so this module invokes the binary with argv it builds itself. run-model.ps1 is left
untouched for manual use.

NETWORKING. llama-server is a native Windows process; the harness runs in WSL2 NAT mode.
WSL cannot reach the Windows loopback, so the server is bound to 0.0.0.0 and reached at the
default-gateway address — which Windows Firewall blocks by default. ``setup/allow_wsl_llamacpp.ps1``
adds the one inbound rule that fixes it; ``run.py doctor`` detects the failure and says so
instead of reporting a generic timeout.
"""

from __future__ import annotations

import re
import shutil
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


def windows_host_ip() -> str | None:
    """The Windows host as seen from WSL2 NAT mode: the default gateway."""
    try:
        out = subprocess.run(
            ["ip", "route", "show", "default"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
    return match.group(1) if match else None


def win_to_wsl(path: str) -> str:
    """``D:\\llm\\...`` -> ``/mnt/d/llm/...``.

    Only the EXECUTABLE needs this: WSL's exec resolves argv[0] itself and cannot open a
    Windows drive path, while every other argument is consumed by the Windows binary and must
    stay in Windows form. Getting this backwards silently hands llama-server a POSIX model
    path it cannot open, which surfaces as a confusing load failure rather than an exec error.
    """
    if len(path) > 1 and path[1] == ":":
        return f"/mnt/{path[0].lower()}/" + path[2:].lstrip("\\/").replace("\\", "/")
    return path


def resolve_host(configured: str) -> str:
    """``host: auto`` -> the Windows gateway IP; anything else is taken literally."""
    if configured and configured != "auto":
        return configured
    host = windows_host_ip()
    if not host:
        raise RuntimeError(
            "Could not determine the Windows host IP from `ip route`. Set "
            "llamacpp.host explicitly in config.yaml."
        )
    return host


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
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        if trim_log_lines:
            self.trim_log(trim_log_lines)

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


def build_argv(cfg: dict[str, Any], model_cfg: dict[str, Any], host: str) -> list[str]:
    """The single place a llama-server command line is constructed.

    Everything that could differ between models and pollute the comparison is read from
    ``config.yaml`` and applied identically; only ``--model`` and the per-model overrides
    explicitly declared in the model's own entry vary.
    """
    llama = cfg["llamacpp"]
    run = cfg["run"]
    models_root = llama["models_root"].rstrip("\\/")
    gguf = f"{models_root}\\{model_cfg['gguf']}"

    argv = [
        win_to_wsl(llama["server_exe"]),   # argv[0] is resolved by WSL, so it must be POSIX
        "-m", gguf,                        # every other arg is parsed by the Windows binary

        "--host", "0.0.0.0",             # WSL reaches the Windows box by IP, never loopback
        "--port", str(llama["port"]),
        "-c", str(run["num_ctx"]),       # fixed at launch under llama.cpp — the whole reason
        "--n-gpu-layers", str(run["n_gpu_layers"]),
        "--flash-attn", "on",            # required to quantize the V cache
        "--cache-type-k", run["cache_type_k"],
        "--cache-type-v", run["cache_type_v"],
        "--jinja",                       # each gguf's own chat template, exactly like the app
        "--parallel", "1",
        "--no-mmap",
        "--no-webui",
        # Debug verbosity is REQUIRED, not optional: it is the only level at which this build
        # logs per-layer device assignment, and without that the harness cannot tell a fully
        # offloaded model from one silently running half on the CPU. Costs a few MB of log.
        "-lv", str(llama.get("log_verbosity", 5)),
    ]
    # MoE expert offload to CPU. A no-op on dense models, so it is applied uniformly rather
    # than conditionally — one fewer axis on which models differ.
    if run.get("n_cpu_moe") is not None:
        argv += ["--n-cpu-moe", str(run["n_cpu_moe"])]
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
    host = host or resolve_host(cfg["llamacpp"].get("host", "auto"))
    argv = build_argv(cfg, model_cfg, host)
    base_url = f"http://{host}:{cfg['llamacpp']['port']}"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-server_{model_cfg['name']}.log"
    log_file = log_path.open("w")
    process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    handle = ServerHandle(process=process, log_path=log_path, argv=argv, base_url=base_url)

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
        f"If the log below shows the server listening, WSL cannot reach it — run "
        f"`python run.py doctor`. Last lines:\n{tail}"
    )
