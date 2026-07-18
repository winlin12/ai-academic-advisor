"""Run orchestration.

Measurement discipline enforced here:
  * one options dict, built once from config, sent identically to every model
  * per model: warmup generations (discarded) before anything is measured
  * VRAM = nvidia-smi delta (baseline before load vs. after warmup); None off-GPU boxes
  * GPU offload: parsed from the server log ("offloaded X/Y layers to GPU") when
    server_log_cmd is configured; /api/ps memory split recorded only as a fallback
    signal and labeled unverified
  * models are unloaded between blocks so the next model starts from a clean card
  * every record carries the static-prompt hash + config snapshot; records with
    different hashes must never be compared

Stage independence: summarization (stage B) is fed rows from the GOLD query, not from
the model's own stage-A SQL — so a bad SQL writer doesn't contaminate the measurement
of the same model's summarization faithfulness.

Stage C (e2e) is the deliberate exception: it feeds the model's OWN stage-A SQL rows
(when stage A produced valid SQL) into the same summarizer, because that — not stage B —
is what a student actually receives from the app. Stage B isolates summarization skill;
stage C measures the whole pipeline. Both are kept, on purpose, as separate metrics.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import scorers
from .db import ro_connect, execute_select
from .ollama_client import GenerationResult, OllamaClient, OllamaError
from .prompts import PromptBuilder, SQL_JSON_SCHEMA

_OFFLOAD_RE = re.compile(r"offload(?:ed|ing)?[^\d]*(\d+)\s*/\s*(\d+)\s+layers to GPU")
WARMUP_PROMPT = "Reply with the single word: ready"


@dataclass
class EvalContext:
    cfg: dict[str, Any]
    root: Path
    client: OllamaClient
    prompts: PromptBuilder
    questions: list[dict[str, Any]]

    @property
    def results_dir(self) -> Path:
        d = self.root / self.cfg["paths"]["results_dir"]
        d.mkdir(exist_ok=True)
        return d


def load_context(root: Path) -> EvalContext:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    questions = yaml.safe_load((root / cfg["paths"]["questions"]).read_text())
    client = OllamaClient(
        cfg["ollama"]["base_url"], timeout_s=cfg["run"].get("request_timeout_s", 300)
    )
    prompts = PromptBuilder(root / cfg["paths"]["schema_sql"])
    return EvalContext(cfg=cfg, root=root, client=client, prompts=prompts, questions=questions)


def _options(cfg: dict[str, Any]) -> dict[str, Any]:
    run = cfg["run"]
    return {
        "num_ctx": run["num_ctx"],
        "temperature": run["temperature"],
        "seed": run["seed"],
        "top_p": run.get("top_p", 0.9),
        "num_predict": run["max_output_tokens"],
    }


# --- environment probes -----------------------------------------------------------------

def gpu_memory_used_mb() -> list[int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return [int(x) for x in out.split()]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None  # no NVIDIA GPU on this box (e.g. authoring on the Mac) — recorded as null


def gpu_offload(ctx: EvalContext, model: str) -> dict[str, Any]:
    """Layer-offload truth from the server log; /api/ps memory split as labeled fallback."""
    result: dict[str, Any] = {"source": "unverified", "layers": None, "fully_offloaded": None}
    cmd = ctx.cfg["ollama"].get("server_log_cmd")
    if cmd:
        try:
            log = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=20
            ).stdout
            matches = _OFFLOAD_RE.findall(log)
            if matches:
                on_gpu, total = map(int, matches[-1])
                result = {
                    "source": "server_log",
                    "layers": f"{on_gpu}/{total}",
                    "fully_offloaded": on_gpu == total,
                }
        except subprocess.SubprocessError:
            pass
    if result["source"] == "unverified":
        for m in ctx.client.ps():
            if m.get("name", "").startswith(model) or m.get("model", "").startswith(model):
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                result = {
                    "source": "api_ps_memory_split (NOT layer truth)",
                    "layers": None,
                    "fully_offloaded": bool(size) and vram >= size,
                }
    return result


def write_meta(ctx: EvalContext, tag: str) -> None:
    meta = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ollama_version": ctx.client.version(),
        "options": _options(ctx.cfg),
        "runs_per_pair": ctx.cfg["run"]["runs_per_pair"],
        "warmup_discard": ctx.cfg["run"]["warmup_discard"],
        "sql_static_hash_v1": ctx.prompts.build_sql_prompt("x")[1],
        "sql_static_hash_v2_structured": ctx.prompts.build_sql_prompt(
            "x", few_shot="v2", structured=True)[1],
    }
    (ctx.results_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] ollama {meta['ollama_version']}, options {meta['options']}")


# --- per-model execution ------------------------------------------------------------------

def _gen_record(model: str, q: dict, run_idx: int, res: GenerationResult,
                static_hash: str, stage: str, mitigated: bool) -> dict[str, Any]:
    return {
        "model": model,
        "question_id": q["id"],
        "category": q["category"],
        "expected_behavior": q["expected_behavior"],
        "stage": stage,
        "run_idx": run_idx,
        "mitigated": mitigated,
        "static_hash": static_hash,
        "ttft_s": res.ttft_s,
        "total_s": res.total_s,
        "eval_count": res.eval_count,
        "prompt_eval_count": res.prompt_eval_count,
        "load_duration_s": res.load_duration_s,
        "output": res.text,
    }


def warmup_model(ctx: EvalContext, model_cfg: dict) -> dict[str, Any]:
    """Load + warm the model; returns environment info measured AFTER warmup."""
    name, think = model_cfg["name"], model_cfg.get("think")
    baseline = gpu_memory_used_mb()
    for i in range(ctx.cfg["run"]["warmup_discard"]):
        ctx.client.generate(
            name, WARMUP_PROMPT, options=_options(ctx.cfg), think=think,
            keep_alive=ctx.cfg["run"]["keep_alive"],
        )
    after = gpu_memory_used_mb()
    vram_delta = (
        [a - b for a, b in zip(after, baseline)] if baseline and after else None
    )
    env = {
        "vram_baseline_mb": baseline,
        "vram_after_warmup_mb": after,
        "vram_delta_mb": vram_delta,
        "offload": gpu_offload(ctx, name),
        "think_setting": think,
    }
    print(f"[warmup] {name}: vram_delta={vram_delta} offload={env['offload']}")
    return env


def run_model(ctx: EvalContext, model_cfg: dict, out, *, mitigate: bool = False) -> None:
    name, think = model_cfg["name"], model_cfg.get("think")
    opts = _options(ctx.cfg)
    keep_alive = ctx.cfg["run"]["keep_alive"]
    few_shot = "v2" if mitigate else "v1"
    structured = mitigate
    conn = ro_connect(ctx.root / ctx.cfg["paths"]["db_file"])

    try:
        env = warmup_model(ctx, model_cfg)
    except OllamaError as exc:
        print(f"[SKIP] {name}: {exc}")
        out.write(json.dumps({"model": name, "stage": "error", "error": str(exc)}) + "\n")
        return
    out.write(json.dumps({"model": name, "stage": "env", **env}) + "\n")

    for q in ctx.questions:
        gold = (q.get("gold_sql") or "").strip() or None
        for run_idx in range(ctx.cfg["run"]["runs_per_pair"]):
            # ---- stage A: text-to-SQL -------------------------------------------
            prompt, static_hash = ctx.prompts.build_sql_prompt(
                q["question"], few_shot=few_shot, structured=structured)
            try:
                res = ctx.client.generate(
                    name, prompt, options=opts, think=think, keep_alive=keep_alive,
                    output_format=SQL_JSON_SCHEMA if structured else None)
            except OllamaError as exc:
                out.write(json.dumps({
                    "model": name, "question_id": q["id"], "stage": "sql",
                    "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
                continue
            parsed = scorers.parse_output(res.text)
            sql_score = scorers.score_sql(conn, parsed.sql, gold)

            # mitigation: ONE retry with the execution error appended
            if mitigate and parsed.action == "sql" and sql_score.valid is False:
                retry_prompt, static_hash = ctx.prompts.build_sql_retry_prompt(
                    q["question"], parsed.sql or "", sql_score.error or "",
                    few_shot=few_shot, structured=structured)
                res2 = ctx.client.generate(
                    name, retry_prompt, options=opts, think=think, keep_alive=keep_alive,
                    output_format=SQL_JSON_SCHEMA)
                parsed2 = scorers.parse_output(res2.text)
                score2 = scorers.score_sql(conn, parsed2.sql, gold)
                res2.total_s += res.total_s  # user-visible cost of the retry path
                res, parsed, sql_score, retried = res2, parsed2, score2, True
            else:
                retried = False

            rec = _gen_record(name, q, run_idx, res, static_hash, "sql", mitigate)
            rec["retried"] = retried
            scorers.record_scores(rec, parsed, sql_score, q["expected_behavior"])
            out.write(json.dumps(rec, default=str) + "\n")

            # ---- stage B: grounded summarization (gold rows => independent) ------
            if gold and q["expected_behavior"] == "answer":
                cols, rows = execute_select(conn, gold)
                sprompt, s_hash = ctx.prompts.build_summary_prompt(q["question"], cols, rows)
                try:
                    sres = ctx.client.generate(
                        name, sprompt, options=opts, think=think, keep_alive=keep_alive)
                except OllamaError as exc:
                    out.write(json.dumps({
                        "model": name, "question_id": q["id"], "stage": "summary",
                        "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
                    continue
                gold_rows_json = json.dumps([dict(zip(cols, r)) for r in rows],
                                            sort_keys=True, default=str)
                srec = _gen_record(name, q, run_idx, sres, s_hash, "summary", mitigate)
                srec["rows_json"] = gold_rows_json
                srec["faithfulness_flags"] = scorers.faithfulness_flags(sres.text, gold_rows_json)
                srec["needs_review"] = True  # every summary is manually graded, no exceptions
                out.write(json.dumps(srec, default=str) + "\n")

                # ---- stage C: end-to-end (model's OWN retrieval => what a student sees) --
                if parsed.action == "sql" and sql_score.valid:
                    e_cols, e_rows = sql_score.columns, sql_score.rows
                    e_prompt, e_hash = ctx.prompts.build_summary_prompt(
                        q["question"], e_cols, e_rows)
                    try:
                        e_res = ctx.client.generate(
                            name, e_prompt, options=opts, think=think, keep_alive=keep_alive)
                    except OllamaError as exc:
                        out.write(json.dumps({
                            "model": name, "question_id": q["id"], "stage": "e2e",
                            "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
                    else:
                        e_rows_json = json.dumps([dict(zip(e_cols, r)) for r in e_rows],
                                                 sort_keys=True, default=str)
                        faith = scorers.faithfulness_flags(e_res.text, e_rows_json)
                        recall = scorers.recall_flags(e_res.text, gold_rows_json)
                        erec = _gen_record(name, q, run_idx, e_res, e_hash, "e2e", mitigate)
                        erec["retrieval_correct"] = sql_score.correct
                        erec["rows_json"] = e_rows_json
                        erec["gold_rows_json"] = gold_rows_json
                        erec["faithfulness_flags"] = faith
                        erec["recall_flags"] = recall
                        erec["e2e_auto_pass"] = bool(sql_score.correct) and not faith and not recall
                        erec["needs_review"] = True
                        out.write(json.dumps(erec, default=str) + "\n")
                else:
                    # No grounded data reached the summarizer at all (bad/no SQL) — this
                    # IS the end-to-end outcome the student would get, not a skipped case.
                    out.write(json.dumps({
                        "model": name, "question_id": q["id"], "category": q["category"],
                        "expected_behavior": q["expected_behavior"], "stage": "e2e",
                        "run_idx": run_idx, "mitigated": mitigate, "action": parsed.action,
                        "retrieval_correct": False, "e2e_auto_pass": False,
                        "e2e_outcome": "no_grounded_data", "needs_review": False}) + "\n")
        out.flush()

    conn.close()
    ctx.client.unload(name)


def run_eval(root: Path, *, models: list[str] | None = None,
             brackets: list[str] | None = None, mitigate: bool = False) -> Path:
    ctx = load_context(root)
    tag = "mitigated" if mitigate else "baseline"
    write_meta(ctx, tag)
    selected = [
        m for m in ctx.cfg["models"]
        if (not models or m["name"] in models)
        and (not brackets or m.get("bracket") in brackets)
    ]
    if not selected:
        raise SystemExit("No models matched the filter.")
    out_path = ctx.results_dir / f"runs_{tag}.jsonl"
    mode = "a" if out_path.exists() else "w"
    print(f"[run] {len(selected)} model(s) -> {out_path} (mode={mode})")
    with out_path.open(mode) as out:
        for m in selected:
            print(f"\n=== {m['name']} ({m.get('bracket')}) ===")
            run_model(ctx, m, out, mitigate=mitigate)
    return out_path
