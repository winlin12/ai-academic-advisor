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


def build_argv(cfg: dict[str, Any], model_cfg: dict[str, Any], host: str) -> list[str]:
    """The single place a llama-server command line is constructed.

    Everything that could differ between models and pollute the comparison is read from
    ``config.yaml`` and applied identically; only ``--model`` and the per-model overrides
    explicitly declared in the model's own entry vary.
    """
    llama = cfg["llamacpp"]
    run = cfg["run"]
    windows = is_windows_exe(llama["server_exe"])
    gguf = to_server_path(Path(llama["models_root"]) / model_cfg["gguf"], windows=windows)

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
