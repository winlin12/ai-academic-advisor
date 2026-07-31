"""MODE C's report. Kept out of ``report.py`` because the record schema is different.

Mode A/B records are pass/fail per plan. A Mode C record is a SURVIVAL OBSERVATION: a case
either converged at attempt k / second t, or it is right-censored at the point the clock or the
attempt budget ran out. Pooling those into the same "% viable" table would either throw away
the censored cases or count them as failures, and both distort the answer in the same
direction. So this module has its own tables and its own estimator.

THE ESTIMATOR. `median attempts to converge` over censored data is not `median(observed
values)`. A case censored at attempt 7 is evidence that the model needed MORE than 7, and
dropping it biases the median down (the slow cases are exactly the ones that get censored),
while scoring it as 7 biases it up in a different way. The Kaplan-Meier product-limit estimator
is the standard instrument for this and it is about twenty lines with no dependencies, so it is
implemented here rather than approximated. When too few cases converge for the survival curve
to reach 0.5, the median is genuinely UNDEFINED and is printed as `>N` rather than invented.

THROUGHPUT IS PRINTED, AND FENCED. `tok/s` sits in its own column with a warning above the
table, because wall-clock convergence time is the product of reasoning quality and hardware
speed and this box's numbers are specific to one card. A model that converges in fewer attempts
but more seconds is a better planner on a slower path; the two columns must never be collapsed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .convergence import CONVERGENCE_CRITERION_CAVEAT

# This arm was briefly called Mode D while it was being built, and records written in that
# window carry `plan_mode_d`. Both are read so a rename does not orphan a completed run. Note
# `results_old8` also holds `plan_mode_c` rows from the REMOVED reference arm — they are a
# different experiment entirely, and what keeps them out is the FILE, not the stage string:
# this report only ever opens `runs_convergence.jsonl`.
_STAGES = ("plan_mode_c", "plan_mode_d")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --- Kaplan-Meier ------------------------------------------------------------------------------


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


def _fmt_quantile(value: float | None, defined: bool, cases: list[dict], key: str,
                  unit: str = "") -> str:
    """A number when the estimator has one, an honest lower bound when it does not."""
    if defined and value is not None:
        return f"{value:g}{unit}"
    if not cases:
        return "—"
    # Undefined: report the largest thing we actually observed, marked as a bound.
    bound = max(c[key] for c in cases)
    return f">{bound:g}{unit}"


# --- tables -------------------------------------------------------------------------------------


def _variant_rows(records: list[dict], variant: str) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("stage") not in _STAGES or rec.get("variant") != variant:
            continue
        by_model.setdefault(rec["model"], []).append(rec)
    return by_model


def _model_stats(cases: list[dict]) -> dict[str, Any]:
    # `expect_unsatisfiable` scenarios are excluded from the headline exactly as in Mode A/B:
    # mi-tight-horizon cannot cover its requirements, and while it CAN converge under this
    # mode's looser criterion, mixing a scenario with a different ceiling into the same median
    # describes neither.
    scored = [c for c in cases if not c.get("expect_unsatisfiable")]
    attempts_obs = [
        (float(c["attempts_to_converge"] if c["converged"] else c["attempts_used"]),
         bool(c["converged"]))
        for c in scored
    ]
    wall_obs = [(float(c["wall_clock_s"]), bool(c["converged"])) for c in scored]

    a50, a50ok = km_quantile(attempts_obs, 0.5)
    a90, a90ok = km_quantile(attempts_obs, 0.9)
    w50, w50ok = km_quantile(wall_obs, 0.5)
    w90, w90ok = km_quantile(wall_obs, 0.9)

    timeouts = [c for c in scored if c.get("censored")]
    tok = [c["tokens_per_s"] for c in scored if c.get("tokens_per_s")]
    return {
        "n": len(scored),
        "converged": sum(1 for c in scored if c["converged"]),
        "attempts_p50": _fmt_quantile(a50, a50ok, scored, "attempts_used"),
        "attempts_p90": _fmt_quantile(a90, a90ok, scored, "attempts_used"),
        "wall_p50": _fmt_quantile(w50, w50ok, scored, "wall_clock_s", "s"),
        "wall_p90": _fmt_quantile(w90, w90ok, scored, "wall_clock_s", "s"),
        "timeouts": len(timeouts),
        "timeout_rate": (len(timeouts) / len(scored)) if scored else 0.0,
        "censor_reasons": {
            reason: sum(1 for c in timeouts if c.get("censor_reason") == reason)
            for reason in sorted({c.get("censor_reason") for c in timeouts} - {None})
        },
        # TOTAL WALL-CLOCK OVER EVERY CASE, including the censored ones and including the
        # by-design-unsatisfiable scenario. This is the apples-to-apples number: "what does it
        # cost to put this model through the whole suite." A median hides the cases that ran to
        # the attempt cap, and those are exactly where a weak model spends its time — a model
        # that converges fast on six scenarios and grinds ten attempts on two can look quicker
        # on the median while costing more end to end.
        "total_wall_s": sum(c["wall_clock_s"] for c in cases),
        "total_attempts": sum(c["attempts_used"] for c in cases),
        "s_per_attempt": (
            round(sum(c["wall_clock_s"] for c in cases)
                  / max(1, sum(c["attempts_used"] for c in cases)), 1)),
        "tok_s": round(sum(tok) / len(tok), 1) if tok else None,
        "strict": sum(1 for c in scored if c.get("converged_strict")),
        "degenerate": sum(1 for c in scored if c.get("degenerate_convergence")),
        "locked_alterations": sum(c.get("locked_alterations_total", 0) for c in scored),
    }


def _table(by_model: dict[str, list[dict]]) -> list[str]:
    if not by_model:
        return ["_No cases recorded for this variant._", ""]
    lines = [
        "| model | n | converged | attempts p50 | attempts p90 | wall p50 | wall p90 | "
        "timeouts | tok/s | strict | degenerate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for model, cases in by_model.items():
        stats = _model_stats(cases)
        rows.append((stats, model))

    # Ranked by how often they converge at all, then by median attempts. Not by wall clock —
    # that is the hardware-confounded axis and it must not decide the ordering.
    #
    # Sorting on `attempts_p50` DIRECTLY would sort its formatted string, putting "10" ahead of
    # "4" and ">6" last regardless of magnitude. The sort key is numeric, and an undefined
    # median (">N") sorts after every real one, which is correct: not knowing the median because
    # too few cases converged is worse than any median you could actually measure.
    def rank(stats: dict[str, Any]) -> tuple:
        text = str(stats["attempts_p50"])
        undefined = text.startswith(">")
        try:
            value = float(text.lstrip(">"))
        except ValueError:
            value, undefined = float("inf"), True
        return (-stats["converged"], undefined, value)

    rows.sort(key=lambda pair: rank(pair[0]))
    for stats, model in rows:
        lines.append(
            f"| {model} | {stats['n']} | {stats['converged']}/{stats['n']} | "
            f"{stats['attempts_p50']} | {stats['attempts_p90']} | "
            f"{stats['wall_p50']} | {stats['wall_p90']} | "
            # The BREAKDOWN, not just the count: "plateau x5" and "timeout x5" are different
            # findings — one is a model that stopped improving, the other a model too slow to
            # finish — and a single "censored" number cannot tell them apart.
            f"{stats['timeouts']} ({stats['timeout_rate']:.0%})"
            + (" " + ", ".join(f"{k} x{v}" for k, v in stats["censor_reasons"].items())
               if stats["censor_reasons"] else "")
            + " | "
            f"{stats['tok_s'] if stats['tok_s'] is not None else '—'} | "
            f"{stats['strict']}/{stats['n']} | {stats['degenerate']} |"
        )
    lines.append("")
    return lines


def _total_time_table(records: list[dict]) -> list[str]:
    """END-TO-END COST PER MODEL. The apples-to-apples column.

    Every other timing number here is conditional on something — a median is over the cases
    that converged, a p90 is over a censored distribution. This one is not: it is the total
    wall-clock to put a model through the entire suite, censored cases and all. It is the
    number that answers "what does running this model actually cost", and it is the honest way
    to compare two models of the same family (gemma4-26b against gemma4-e4b) because both sat
    on the same card, ran the same scenarios and were held to the same criterion.

    It still is NOT a quality ranking, and the two columns beside it say why: a model that
    converges on every scenario in two attempts can lose this column to one that gives up
    early. Read `converged` first, `total` second.
    """
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("stage") not in _STAGES:
            continue
        by_model.setdefault(rec["model"], []).append(rec)
    if not by_model:
        return []

    rows = []
    for model, cases in by_model.items():
        stats = _model_stats(cases)
        rows.append((stats["total_wall_s"], model, stats, len(cases)))
    rows.sort()

    out = [
        "## Total time to run every case",
        "",
        "**The apples-to-apples number.** Total wall-clock for the whole suite per model — "
        "every scenario, every attempt, censored cases included. Same card, same scenarios, "
        "same criterion, so two models of one family are directly comparable here.",
        "",
        "| model | cases | converged | total time | total attempts | s/attempt | tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for total, model, stats, n in rows:
        mins = total / 60.0
        out.append(
            f"| {model} | {n} | {stats['converged']}/{stats['n']} | "
            f"**{total:.0f}s** ({mins:.1f} min) | {stats['total_attempts']} | "
            f"{stats['s_per_attempt']}s | "
            f"{stats['tok_s'] if stats['tok_s'] is not None else '—'} |"
        )
    grand = sum(r[0] for r in rows)
    out += [
        "",
        f"Whole run: **{grand:.0f}s ({grand / 60:.1f} min)** across "
        f"{sum(r[3] for r in rows)} case(s).",
        "",
        "> `total time` is cost, not quality. A model that gives up early can post a lower "
        "total than one that converges on everything — read it next to `converged`, and read "
        "`s/attempt` to separate 'fewer attempts' from 'faster attempts'.",
        "",
    ]
    return out


def _ratchet_guard(by_model: dict[str, list[dict]]) -> list[str]:
    """The repair variant's core claim, checked rather than asserted.

    `frozen_by_attempt` is the count of validated placements carried into the next attempt. It
    must never DECREASE — that is what "good work is never lost" means, and it is the whole
    difference between this variant and the other two. If it falls, the freeze has a bug and
    every number in the table above is describing something other than a ratchet. Printing the
    check is cheaper than trusting it.
    """
    violations: list[str] = []
    removed = frozen = cases = 0
    unrepairable: list[str] = []
    for model, records in by_model.items():
        for rec in records:
            trace = rec.get("frozen_by_attempt") or []
            releases = rec.get("released_by_attempt") or []
            cases += 1
            removed += rec.get("repair_removed_total", 0)
            if trace:
                frozen = max(frozen, trace[-1])
            # A fall is legitimate exactly when a deadlock release happened on that step.
            # Anything else means the freeze lost validated work, which is the one thing this
            # variant promises not to do.
            # INDEX i+1, NOT i. Both lists are appended once per attempt, and within an attempt
            # the release happens BEFORE the freeze — so the drop from trace[i] to trace[i+1] is
            # explained by the release recorded on attempt i+1. Reading releases[i] here flagged
            # every legitimate deadlock break as lost work.
            for i, (a, b) in enumerate(zip(trace, trace[1:])):
                if b < a and not (i + 1 < len(releases) and releases[i + 1]):
                    violations.append(
                        f"{model}/{rec['scenario']}: frozen fell {a} -> {b} "
                        f"with no deadlock release")
            for item in rec.get("repair_unrepairable") or []:
                unrepairable.append(f"{model}/{rec['scenario']}: {item}")

    out = ["**Ratchet check.** "]
    if violations:
        out = [
            f"> ❌ **The ratchet broke in {len(violations)} case(s)** — validated placements "
            f"were lost between attempts, so this variant is not doing what it claims:", "",
        ] + [f"> - {v}" for v in violations[:10]] + [""]
    else:
        released = sum(r.get("repair_released_total", 0)
                       for rs in by_model.values() for r in rs)
        out = [
            f"**Ratchet holds:** validated placements never decreased across any of {cases} "
            f"case(s) except where a deadlock release explains it. {removed} placement(s) were "
            f"removed deterministically by the harness rather than by a model round trip; "
            f"{released} were released after coverage stalled (legal-but-too-late placements "
            f"that were blocking the unmet requirements).", "",
        ]
    if unrepairable:
        out += [
            f"**{len(unrepairable)} violation(s) could not be repaired** — the only fix would "
            f"have been to move a course the STUDENT locked, which the harness refuses to do:",
            "",
        ] + [f"- {u}" for u in unrepairable[:10]] + [""]
    return out


def _progress_table(records: list[dict]) -> list[str]:
    """Is a model making progress, or just resampling?

    The single most diagnostic thing this mode produces, and it needs the per-attempt log to
    exist. A violation trace of 7 -> 4 -> 1 is a model reading the feedback; 7 -> 6 -> 8 -> 7 is
    a model rerolling dice with extra steps. The blind variant is expected to look like the
    second — that is its purpose — so the comparison to make is feedback against blind for the
    SAME model.
    """
    lines = [
        "| model | variant | scenario | violations by attempt | coverage by attempt | outcome |",
        "|---|---|---|---|---|---|",
    ]
    any_row = False
    for rec in records:
        if rec.get("stage") not in _STAGES:
            continue
        trace = rec.get("violations_by_attempt") or []
        if len(trace) < 2:
            continue  # converged first try, or never got off the ground — nothing to trace
        any_row = True
        cov = " -> ".join(f"{c:.0%}" for c in (rec.get("coverage_by_attempt") or []))
        outcome = (f"converged @ {rec['attempts_to_converge']}" if rec["converged"]
                   else f"censored ({rec.get('censor_reason')})")
        lines.append(
            f"| {rec['model']} | {rec['variant']} | {rec['scenario']} | "
            f"{' -> '.join(str(v) for v in trace)} | {cov} | {outcome} |"
        )
    if not any_row:
        return ["_Every case converged or failed on its first attempt — no traces to show._", ""]
    lines.append("")
    return lines


def generate_report(root: Path, tag: str = "convergence") -> Path:
    results = root / "results"
    records = _load(results / f"runs_{tag}.jsonl")
    meta_path = results / f"meta_{tag}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out: list[str] = [
        "# Mode C — retry to convergence",
        "",
        "How many attempts, and how long, until a model lands a plan with no errors in it.",
        "",
    ]

    if not records:
        out += ["_No Mode C records found. Run `python run.py converge`._", ""]
        (results / "convergence_report.md").write_text("\n".join(out))
        return results / "convergence_report.md"

    conv = meta.get("convergence", {})
    out += [
        "## Run settings",
        "",
        f"- timeout: **{conv.get('timeout_s')} s per case**, "
        f"max attempts: **{conv.get('max_attempts')}**",
        f"- convergence criterion: `{', '.join(conv.get('violation_classes') or [])}` "
        f"= 0, over filled slots, no coverage condition",
        f"- feedback list: "
        + (f"top {conv['feedback_top_n']}" if conv.get("feedback_top_n")
           else "full (every violation shown)"),
        f"- plateau stop: after **{conv.get('plateau_patience')}** attempts with no change in "
        f"(violations, coverage), for {', '.join(conv.get('plateau_variants') or []) or 'no'} "
        f"variant(s) — recorded as censored, never as a failure",
        f"- excluded prereq edges: "
        f"{', '.join(conv.get('excluded_prereq_edges') or []) or 'none'}",
        f"- fixture: `{meta.get('fixture', {}).get('hash')}` "
        f"(verified: {meta.get('fixture', {}).get('verified')})",
        "",
        "> **Read `converged` next to `strict` and `degenerate`.**",
        "",
    ] + [f"> {line}" for line in CONVERGENCE_CRITERION_CAVEAT.splitlines()] + [""]

    hashes = {r.get("static_hash") for r in records if r.get("static_hash")}
    if len(hashes) > 1:
        out += [
            f"> ⚠️ **{len(hashes)} different static prompt hashes in this file "
            f"({', '.join(sorted(h[:8] for h in hashes))}). These records describe different "
            f"experiments and must not be pooled.**", "",
        ]
    fixtures = {r.get("fixture_hash") for r in records if r.get("fixture_hash")}
    if len(fixtures) > 1:
        out += [
            f"> ⚠️ **{len(fixtures)} different fixture hashes. The scoring authority changed "
            f"mid-file; these rows are not comparable.**", "",
        ]

    for variant, blurb in (
        ("repair", "The harness deletes violating placements itself and freezes everything "
                   "that validates; the model only fills what is missing. **Scored STRICTLY** "
                   "(clean AND fully covered) — the repair step makes 'clean' free, so the "
                   "loose criterion would score every model as converging on attempt 1. "
                   "`converged` and `strict` are therefore the same column here."),
        ("feedback", "Validator violations are fed back into the next attempt. "
                     "**Self-correction.**"),
        ("blind", "Same prompt, new sample, no information about what went wrong. "
                  "**Raw variance.**"),
    ):
        rows = _variant_rows(records, variant)
        out += [f"## Variant: {variant}", "", blurb, ""]
        out += _table(rows)
        if variant == "repair" and rows:
            out += _ratchet_guard(rows)

    out += _total_time_table(records)
    out += [
        "## Throughput is not quality",
        "",
        "`tok/s` above is generation throughput on **this box only** (see the env rows in the "
        "baseline run). A model needing 3 attempts at 23 tok/s can beat a model needing 1 "
        "attempt at 6 tok/s on wall-clock while being the worse planner. `attempts p50` is the "
        "reasoning column; `wall p50` is the reasoning column multiplied by this card. Rank on "
        "attempts; quote wall-clock only about this hardware.",
        "",
        "## Progress traces",
        "",
        "Violations per attempt, in order. A descending trace is self-correction; a flat or "
        "bouncing one is resampling.",
        "",
    ]
    out += _progress_table(records)

    path = results / "convergence_report.md"
    path.write_text("\n".join(out))
    return path
