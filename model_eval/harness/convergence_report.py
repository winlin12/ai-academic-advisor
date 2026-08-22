"""Mode C's report: what the repair passes bought over the model's first attempt.

MODE C IS MODE B PLUS FEEDBACK (2026-08-17), and both come out of ONE generation pass —
`plan_mode_b` is attempt 1 exactly as the model produced it unaided, `plan_mode_c` is where
that same plan ended up after the validator reported what was wrong and the model had up to
`run.repair_passes` more tries. So every arrow in these tables is one plan before and after,
not two independent samples of the same prompt, which is what makes the delta an effect.

WHAT THE VALIDATOR REPORTS, and the third item is new: hard violations, unmet requirements,
and uneven credit spread. Spread was invisible to the loop until now, which made the repair
passes unable to fix the one defect they were most often needed for — a plan can be legal and
fully covered while handing the student eighteen credits one term and three the next.

TWO EXPERIMENTS THIS FILE HAS OUTLIVED, both worth remembering:

  * A RETRY-TO-CONVERGENCE mode with its own prompt variants (blind / feedback / repair) on
    top of a deterministic auto-repair pass. That was the right instrument when a single fused
    call had to choose courses AND sequence them and usually failed at both; with selection
    deterministic it had nothing left to repair. Its Kaplan-Meier estimator survives below,
    because censoring is still the correct statistics for "attempts until an event, when some
    cases never reach it".
  * A REASONING ARM — Mode B with thinking forced on. Measured on qwen3.8-27b: ~2x latency for
    mean credit spread 13.8 -> 14.4, over-ask 1.4 -> 1.6 and viability 5/5 -> 3/5. Feedback
    earns its tokens on the same model and the same program; thinking did not.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .convergence import CONVERGENCE_CRITERION_CAVEAT

# Stage names the survival helpers accept. `plan_mode_d` is here only so an archived run from
# the two days the retry loop had its own mode still renders rather than raising.
_STAGES = ("plan_mode_c", "plan_mode_d")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rate(values: list[bool]) -> str:
    return f"{(sum(1 for v in values if v) / len(values)):.0%}" if values else "—"


def _mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def _pairs(baseline: list[dict], mode_c: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """(model -> {"b": rows, "c": rows}) over the (model, scenario) pairs present in BOTH.

    Restricted to pairs on purpose: a model whose Mode C run errored and whose Mode B run did
    not would otherwise shift its own delta by dropping its hardest scenario from one side.
    """
    keyed_b = {(r.get("model"), r.get("scenario")): r for r in baseline
               if r.get("stage") == "plan_mode_b" and not r.get("error")}
    keyed_c = {(r.get("model"), r.get("scenario")): r for r in mode_c
               if r.get("stage") == "plan_mode_c" and not r.get("error")}
    shared = sorted(set(keyed_b) & set(keyed_c))
    out: dict[str, dict[str, list[dict]]] = {}
    for key in shared:
        bucket = out.setdefault(key[0], {"b": [], "c": []})
        bucket["b"].append(keyed_b[key])
        bucket["c"].append(keyed_c[key])
    return out


def _delta_table(pairs: dict[str, dict[str, list[dict]]]) -> list[str]:
    lines = [
        "| Model | Fixed by feedback | PLAN_VIABLE 1st → final | Coverage 1st → final | "
        "Violations 1st → final | Spread 1st → final |",
        "|---|---|---|---|---|---|",
    ]
    for model, sides in sorted(pairs.items()):
        b, c = sides["b"], sides["c"]
        # "Fixed by feedback" means a LATER attempt is the one that reached the report —
        # which is `best_attempt`, not `converged_at`. Convergence is the strict bar (nothing
        # left to report at all) and a plan can improve a great deal without reaching it.
        improved = _rate([bool((r.get("best_attempt") or 1) > 1) for r in c])
        vb, vc = _rate([bool(r.get("plan_viable")) for r in b]), _rate(
            [bool(r.get("plan_viable")) for r in c])
        cb, cc = _mean([r.get("requirement_coverage") for r in b]), _mean(
            [r.get("requirement_coverage") for r in c])
        nb, nc = _mean([len(r.get("violations") or []) for r in b]), _mean(
            [len(r.get("violations") or []) for r in c])
        sb, sc = _mean([r.get("credit_spread") for r in b]), _mean(
            [r.get("credit_spread") for r in c])
        cov = (f"{cb:.0%} → {cc:.0%}" if cb is not None and cc is not None else "—")
        vio = (f"{nb:.1f} → {nc:.1f}" if nb is not None and nc is not None else "—")
        sec = (f"{sb:.1f} → {sc:.1f}" if sb is not None and sc is not None else "—")
        lines.append(f"| `{model}` | {improved} | {vb} → {vc} | {cov} | {vio} | {sec} |")
    return lines


def mode_c_lines(results: Path, tag: str = "convergence") -> list[str]:
    """Mode C's section: what the repair passes bought over the model's first attempt.

    BOTH SIDES COME OUT OF ONE FILE AND ONE GENERATION now — `plan_mode_b` is attempt 1 and
    `plan_mode_c` is where it finished, so every pair below is literally the same plan before
    and after feedback rather than two independent samples of the same prompt.
    """
    baseline = _load(results / "runs_baseline.jsonl")
    mode_c = baseline
    if not mode_c:
        return ["_No Mode C records. Mode C is produced by the `plan_b` task — one pass "
                "writes both — so a run with plan_b in its tasks has it._", ""]
    pairs = _pairs(baseline, mode_c)
    out = [
        "Mode C is Mode B plus up to `run.repair_passes` revisions, each shown what the "
        "validator found wrong — violations, unmet requirements, and now uneven credit spread. "
        "Mode B is the first attempt of the same run, so each arrow is one plan before and "
        "after feedback.",
        "",
    ]
    if not pairs:
        return out + ["_Mode C records exist but no (model, scenario) pair has both halves "
                      "yet, so there is nothing to compare._", ""]
    out += _delta_table(pairs)

    cases = [r for r in mode_c if r.get("stage") == "plan_mode_c" and not r.get("error")]
    if cases:
        by_model: dict[str, list[dict]] = {}
        for case in cases:
            by_model.setdefault(case["model"], []).append(case)
        out += ["", "**What the loop cost.** `Clean` is the share of cases the validator ran "
                "out of complaints about — legal, fully covered AND evenly spread, which is a "
                "stricter bar than PLAN_VIABLE above. `Kept attempt` is which attempt actually "
                "reached this report, since the loop keeps its best state rather than its "
                "last.", ""]
        out += _effort_table(by_model)
        out += _attempt_progress(cases)
    out += ["", CONVERGENCE_CRITERION_CAVEAT, ""]
    return out


def mode_c_cross_school_lines(
    school_dirs: list[tuple[str, Path]], tag: str = "convergence"
) -> list[str]:
    """The same paired comparison, pooled across every school in the sweep."""
    baseline: list[dict] = []
    for slug, directory in school_dirs:
        for rec in _load(directory / "runs_baseline.jsonl"):
            rec["scenario"] = f"{slug}:{rec.get('scenario')}"
            baseline.append(rec)
    mode_c = baseline
    if not mode_c:
        return ["_No Mode C records for any school._", ""]
    pairs = _pairs(baseline, mode_c)
    out = [
        "First attempt vs. final, pooled across every school. Scenario keys are prefixed with "
        "the "
        "school so a pair is only ever matched within its own program.",
        "",
    ]
    if not pairs:
        return out + ["_No (model, scenario) pair has both halves yet._", ""]
    out += _delta_table(pairs)
    cases = [r for r in mode_c if r.get("stage") == "plan_mode_c" and not r.get("error")]
    if cases:
        by_model: dict[str, list[dict]] = {}
        for case in cases:
            by_model.setdefault(case["model"], []).append(case)
        out += ["", "**What the loop cost, pooled.** Same columns as a single program's "
                "report; `Clean` is the strict bar (no findings left at all).", ""]
        out += _effort_table(by_model)
    return out + ["", CONVERGENCE_CRITERION_CAVEAT, ""]

# =============================================================================================
# EFFORT — what the repair loop cost, and which attempt actually paid
# =============================================================================================
#
# CENSORING IS STILL THE STATISTICS. A case that used its whole budget without going clean has
# not shown that it needs `repair_passes + 1` attempts; it has shown it needs MORE than that.
# Dropping those cases biases the median down (the stubborn ones are exactly the ones that get
# censored) and scoring them at the budget biases it up, so `attempts to clean` is a
# Kaplan-Meier quantile over right-censored observations and prints as `>3` when the estimator
# is undefined past the budget. This machinery is inherited from the retry experiment that used
# to be its own mode; only what it reads changed.
#
# `KEPT ATTEMPT` IS THE NEW COLUMN AND THE MOST DIAGNOSTIC ONE. The loop reports its BEST state,
# not its last (see `runner._run_staged_plan`'s guard), so this is the attempt whose plan
# actually reached the report. Mean 1.0 means every revision was a waste of tokens; a mean
# above 1 with a low clean rate means the passes are helping without finishing the job.


def km_quantile(
    observations: Iterable[tuple[float, bool]], q: float
) -> tuple[float | None, bool]:
    """Kaplan-Meier quantile over (time, event) pairs. ``event=True`` means converged.

    Returns (value, defined). When the survival curve never falls to ``1 - q`` — too many
    censored cases — the quantile is undefined and the caller prints a lower bound instead of
    a number, because "we never saw half of them converge" is a real answer and inventing a
    median for it is not.
    """
    data = sorted(observations, key=lambda pair: pair[0])
    if not data:
        return None, False
    n_at_risk = len(data)
    survival = 1.0
    target = 1.0 - q
    index = 0
    while index < len(data):
        t = data[index][0]
        tied = [pair for pair in data if pair[0] == t]
        events = sum(1 for _, event in tied if event)
        if events and n_at_risk > 0:
            survival *= (1 - events / n_at_risk)
            if survival <= target + 1e-12:
                return t, True
        n_at_risk -= len(tied)
        index += len(tied)
    return None, False


def _effort_table(by_model: dict[str, list[dict]]) -> list[str]:
    lines = [
        "| Model | Cases | Clean | Attempts to clean (p50 / p90) | Kept attempt | "
        "Improved by feedback | Median s | Tokens/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, cases in sorted(by_model.items()):
        obs = [(float(c.get("attempts_used") or 0), bool(c.get("converged_at"))) for c in cases]
        p50, ok50 = km_quantile(obs, 0.5)
        p90, ok90 = km_quantile(obs, 0.9)
        budget = max((c.get("attempts_used") or 0) for c in cases)
        kept = [c.get("best_attempt") for c in cases if c.get("best_attempt")]
        improved = [bool((c.get("best_attempt") or 1) > 1) for c in cases]
        secs = [c.get("total_s") for c in cases if c.get("total_s")]
        tps = [c["eval_count"] / c["total_s"] for c in cases
               if c.get("eval_count") and c.get("total_s")]
        lines.append(
            f"| `{model}` | {len(cases)} "
            f"| {_rate([bool(c.get('converged_at')) for c in cases])} "
            f"| {(f'{p50:.0f}' if ok50 else f'>{budget:.0f}')} / "
            f"{(f'{p90:.0f}' if ok90 else f'>{budget:.0f}')} "
            f"| {(f'{_mean(kept):.1f}' if kept else '—')} "
            f"| {_rate(improved)} "
            f"| {(f'{_mean(secs):.0f}' if secs else '—')} "
            f"| {(f'{_mean(tps):.1f}' if tps else '—')} |"
        )
    return lines


def _attempt_progress(cases: list[dict]) -> list[str]:
    """Per-attempt coverage and spread, so a loop that is making progress is distinguishable
    from one resampling noise. Truncated — this is a trace, not a table to read end to end."""
    rows = [c for c in cases if (c.get("attempts") or [])][:12]
    if not rows:
        return []
    lines = ["", "**Attempt traces** (coverage / outstanding findings, per attempt):", "",
             "| Model | Scenario | Trace | Kept |", "|---|---|---|---|"]
    for case in rows:
        trace = " → ".join(
            f"{(a.get('coverage') or 0):.0%}/{a.get('findings', '?')}"
            for a in case["attempts"])
        lines.append(f"| `{case.get('model')}` | {case.get('scenario')} | {trace} "
                     f"| {case.get('best_attempt') or '—'} |")
    return lines
