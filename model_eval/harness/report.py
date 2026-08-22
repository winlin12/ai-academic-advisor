"""Report generation. Plan-of-study first, everything else after.

No composite score exists. The report prints the metrics that are automatic and decidable
(plan viability, violation mix, requirement coverage, structured-output success) as tables,
prints the heuristic ones as clearly labelled heuristics, and routes every free-text answer
into a manual review queue.

Guards that refuse to produce a number rather than produce a misleading one:
  * mixed static-prompt hashes  -> those records came from different prompts
  * mixed fixture hashes        -> the scoring authority changed mid-run
  * unverified fixture          -> banner, because prereqs/terms in it are hand-written
  * duplicate model entries     -> a model listed twice pools two conditions into one row
  * server n_ctx != config      -> that model was not compared at the same context size
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .convergence_report import mode_c_cross_school_lines, mode_c_lines


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _count(value: Any) -> int:
    """`duplicates_removed` as a number, whichever shape the record stores it in.

    It was a LIST of the deleted course codes until 2026-08-13 and is a COUNT since — the
    staged pipeline's `_score_semesters` returns the count because nothing downstream reads the
    codes. Records written on either side of that change live in the same `results_old/`, and a
    report that crashes on the older shape makes the archive unreadable for no benefit.
    """
    if isinstance(value, int):
        return value
    return len(value or [])


def _rate(values: list[bool]) -> str:
    if not values:
        return "—"
    return f"{sum(1 for v in values if v) / len(values):.0%}"


def _rate_with_range(by_run: dict[int, list[bool]]) -> str:
    """overall (min–max across replicates). A wide range means the model is unstable at this
    temperature, and single-run comparisons of it are meaningless."""
    flat = [v for values in by_run.values() for v in values]
    if not flat:
        return "—"
    overall = sum(1 for v in flat if v) / len(flat)
    per_run = [sum(1 for v in vs if v) / len(vs) for vs in by_run.values() if vs]
    if len(per_run) < 2:
        return f"{overall:.0%}"
    return f"{overall:.0%} ({min(per_run):.0%}–{max(per_run):.0%})"


def _group(records: list[dict], stage: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("stage") == stage and not rec.get("error"):
            out[rec["model"]].append(rec)
    return out


def _by_run(recs: list[dict], key: str) -> dict[int, list[bool]]:
    out: dict[int, list[bool]] = defaultdict(list)
    for rec in recs:
        if rec.get(key) is not None:
            out[rec.get("run_idx", 0)].append(bool(rec[key]))
    return out


def _consistency(recs: list[dict], item_key: str, metric: str) -> str:
    """Fraction of items where every replicate agreed. Low consistency + small gaps = noise."""
    by_item: dict[Any, list[bool]] = defaultdict(list)
    for rec in recs:
        if rec.get(metric) is not None:
            by_item[rec.get(item_key)].append(bool(rec[metric]))
    multi = [v for v in by_item.values() if len(v) > 1]
    if not multi:
        return "—"
    agreed = sum(1 for v in multi if all(v) or not any(v))
    return f"{agreed / len(multi):.0%}"


def _median(values: list[Any]) -> str:
    clean = [v for v in values if isinstance(v, (int, float))]
    return f"{statistics.median(clean):.1f}" if clean else "—"


def _numbers(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def _stat(values: list[Any], kind: str, places: int = 1) -> str:
    """p50 / mean / p95 over whatever survived as a number.

    p95 is nearest-rank rather than interpolated, so the value printed is one that was
    actually observed: with 24 samples per (model, stage) an interpolated p95 is a number
    no student ever waited.
    """
    clean = sorted(_numbers(values))
    if not clean:
        return "—"
    if kind == "p50":
        value = statistics.median(clean)
    elif kind == "mean":
        value = statistics.fmean(clean)
    elif kind == "p95":
        value = clean[min(len(clean) - 1, int(0.95 * len(clean)))]
    else:
        raise ValueError(kind)
    return f"{value:.{places}f}"


# --- sections ---------------------------------------------------------------------------------


def _satisfiable(recs: list[dict]) -> list[dict]:
    """Drop scenarios flagged `expect_unsatisfiable` in the fixture.

    Those are 0% PLAN_VIABLE for every model by construction — no legal plan covers every
    requirement in the horizon — so pooling them into the headline rate would put a floor on
    the metric that has nothing to do with model quality. They are reported separately, on
    what they actually test: whether the model reports honestly what didn't fit instead of
    inventing a semester or blowing the credit cap.
    """
    return [r for r in recs if not r.get("expect_unsatisfiable")]


def _plan_section(records: list[dict], mode: str, title: str, note: str) -> list[str]:
    groups = _group(records, mode)
    if not groups:
        return []
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines += [
        "| Model | PLAN_VIABLE | Structure OK | Req. coverage | Ask honoured | Violations/plan | Idle cr | Heavy major terms | Over-ask terms | Sem used | Cr spread | Consistency | Median s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows: list[tuple[float, str]] = []
    for model, all_recs in groups.items():
        recs = _satisfiable(all_recs) or all_recs
        viable = _by_run(recs, "plan_viable")
        # A plan can be perfectly legal and still ignore what the student asked for — the
        # `mi-no-filler` scenario is exactly that case, and PLAN_VIABLE cannot see it.
        asserts: list[bool] = []
        for rec in recs:
            asserts.extend((rec.get("assertions_passed") or {}).values())
        coverage = [r["requirement_coverage"] for r in recs
                    if r.get("requirement_coverage") is not None]
        violations = [len(r.get("violations") or []) for r in recs]
        # Read Idle cr NEXT TO Violations/plan, never on its own. A model that schedules almost
        # nothing wins the violations column by default; this is the column that shows the bill.
        idle = [r["idle_credits"] for r in recs if r.get("idle_credits") is not None]
        # SOFT ONLY — never in Violations/plan (see `plan_scorers.py`'s module docstring: a
        # heavy major-course term stopped gating PLAN_VIABLE 2026-08-07, same call as the
        # credit cap below). This is the term a student would call a bad semester but the
        # registrar would still let them register for. Per plan, so it is readable next to
        # Violations/plan.
        soft = [r["soft_major_overloads"] for r in recs
                if r.get("soft_major_overloads") is not None]
        # The credit half of the same idea, added 2026-08-02 when the hard `credit_cap_violation`
        # threshold moved from the student's number to the registrar's 18. Without this column a
        # term over what the student asked for would vanish from the report entirely for the six
        # scenarios that carry no `max_credits_at_most` assertion — it would be under the hard
        # cap (so not in Violations/plan) and unasserted (so not in Ask honoured). It is a
        # preference breach, not a legality one, which is exactly why it needs its own column
        # instead of being folded back into either of those.
        over_ask = [r["soft_credit_overages"] for r in recs
                    if r.get("soft_credit_overages") is not None]
        # Distribution, the pair the credit TARGET was added to move. "Sem used" is
        # semesters-with-courses over semesters-available; "Cr spread" is max-min across
        # non-empty terms. Both diagnostic — a plan that finishes early is not illegal — but
        # together they are what shows whether a target produced a balanced plan or just a
        # front-loaded one with the same violation count.
        used = [r["semesters_used"] for r in recs if r.get("semesters_used")]
        avail = [r["semesters_available"] for r in recs if r.get("semesters_available")]
        spread = [r["credit_spread"] for r in recs if r.get("credit_spread") is not None]
        flat = [v for vs in viable.values() for v in vs]
        cov_cell = f"{(sum(coverage) / len(coverage)):.0%}" if coverage else "—"
        vio_cell = f"{(sum(violations) / len(violations)):.1f}" if violations else "—"
        idle_cell = f"{(sum(idle) / len(idle)):.0f}" if idle else "—"
        soft_cell = f"{(sum(soft) / len(soft)):.1f}" if soft else "—"
        over_ask_cell = f"{(sum(over_ask) / len(over_ask)):.1f}" if over_ask else "—"
        used_cell = (f"{sum(used) / len(used):.1f}/{sum(avail) / len(avail):.1f}"
                     if used and avail else "—")
        spread_cell = f"{(sum(spread) / len(spread)):.1f}" if spread else "—"
        rows.append((
            (sum(1 for v in flat if v) / len(flat)) if flat else -1.0,
            f"| `{model}` | {_rate_with_range(viable)} | "
            f"{_rate([bool(r.get('structure_ok')) for r in recs])} | {cov_cell} | "
            f"{_rate(asserts)} | {vio_cell} | "
            f"{idle_cell} | {soft_cell} | {over_ask_cell} | {used_cell} | {spread_cell} | "
            f"{_consistency(recs, 'scenario', 'plan_viable')} | "
            f"{_median([r.get('total_s') for r in recs])} |",
        ))
    lines += [row for _, row in sorted(rows, key=lambda item: item[0], reverse=True)]

    # WHY plans fail is more actionable than how often.
    lines += ["", "**Where plans break** (violation counts across ALL runs, including the "
              "unsatisfiable scenario — a violation is a violation regardless of whether full "
              "coverage was reachable). The first three columns are genuine registration walls: "
              "term-offering, credit-cap, and major-overload breaches are gone from this table "
              "entirely, not just softened — see `plan_scorers.py`'s module docstring for why. "
              "`duplicates removed` is NOT in that count and does not touch PLAN_VIABLE above — "
              "see the same docstring's 2026-08-07 note: a repeated or already-completed course "
              "is deleted before scoring ever runs, tracked here so it stays visible instead of "
              "silently vanishing:",
              "", "| Model | prereq | coreq | hallucinated | duplicates removed |",
              "|---|---|---|---|---|"]
    for model, recs in groups.items():
        counts: Counter = Counter()
        duplicates = 0
        for rec in recs:
            counts.update(rec.get("violation_counts") or {})
            duplicates += _count(rec.get("duplicates_removed"))
        lines.append(
            f"| `{model}` | {counts.get('prereq_violation', 0)} "
            f"| {counts.get('coreq_violation', 0)} "
            f"| {counts.get('hallucinated_course', 0)} "
            f"| {duplicates} |"
        )
    return lines + [""]


def _mode_a_extra(records: list[dict]) -> list[str]:
    """Mode A's own question: did it find what was wrong with the student's schedule?

    THE ANSWER KEY IS KNOWN. `fixtures.student_draft_schedule` damages a legal plan in two
    specific ways — courses deleted, one course pushed ahead of its prerequisites — seeded off
    the scenario id, so every model audits the identical draft. These are therefore not
    heuristics: `Found missing` is the share of the deleted courses the model actually named,
    and `False alarms` counts courses it called missing that were never removed.

    The columns here used to describe the revise-plan proposal Mode A emitted before 2026-08-13,
    when a deterministic planner rebuilt the schedule afterwards and all seven models scored
    identically because the model's answer barely reached the plan. See `runner.run_mode_a`.
    """
    groups = _group(records, "plan_mode_a")
    if not groups:
        return []
    lines = [
        "**Did it find the damage?** Each student's draft is the deterministic planner's own "
        "legal plan with courses removed and one course moved ahead of its prerequisites. The "
        "model is scored against that answer key, not against its own account of itself.",
        "",
        "| Model | Found missing | False alarms | Moves fixed | Final plan viable | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for model, recs in groups.items():
        rates = [r["found_missing_rate"] for r in recs
                 if r.get("found_missing_rate") is not None]
        moves = [bool(r["fixed_moves"]) for r in recs if r.get("fixed_moves") is not None]
        coverage = [r["requirement_coverage"] for r in recs
                    if r.get("requirement_coverage") is not None]
        found_cell = f"{(sum(rates) / len(rates)):.0%}" if rates else "—"
        coverage_cell = f"{(sum(coverage) / len(coverage)):.0%}" if coverage else "—"
        lines.append(
            f"| `{model}` | {found_cell} "
            f"| {_stat([len(r.get('false_missing') or []) for r in recs], 'mean')} "
            f"| {_rate(moves) if moves else '—'} "
            f"| {_rate_with_range(_by_run(recs, 'plan_viable'))} "
            f"| {coverage_cell} |"
        )
    return lines + [""]


def _selection_section(records: list[dict]) -> list[str]:
    """Stage 1 of Mode B/C, scored on its own.

    THE POINT OF SPLITTING THE STAGES IS THIS TABLE. `Selection coverage` is how much of the
    degree the chosen course SET covers with ordering ignored entirely; the plan coverage in
    the Mode B table above is what survived scheduling. A model whose selection is 100% and
    whose plan is 80% lost the difference in stage two; a model at 55% in both never had a
    chance at stage two, and no amount of prompt work on the scheduler will help it.
    """
    groups = {m: [r for r in recs if r.get("selection_coverage") is not None]
              for m, recs in _group(records, "plan_mode_b").items()}
    groups = {m: recs for m, recs in groups.items() if recs}
    if not groups:
        return []
    lines = ["### Stage 1 — course selection (Mode B/C's first pass)", "",
             "Requirement groups in, course codes out, no scheduling. Compare `Selection "
             "coverage` against Mode B's plan coverage above: the gap between them is what the "
             "ordering stage lost.", "",
             "`Prereqs supplied` is the repair the pipeline had to make: courses the model's "
             "selection needed but never chose, because they satisfy no requirement group and "
             "are pure prerequisites (see `runner._selection_stage`). Zero is a clean "
             "selection; a large number means the model read the requirement list and not the "
             "chains underneath it.", "",
             "| Model | Parsed | Courses chosen | Prereqs supplied | Selection coverage | "
             "Plan coverage | Lost in ordering |",
             "|---|---|---|---|---|---|---|"]
    for model, recs in groups.items():
        selection = [r["selection_coverage"] for r in recs]
        plan = [r["requirement_coverage"] for r in recs
                if r.get("requirement_coverage") is not None]
        sel_mean = sum(selection) / len(selection)
        plan_mean = (sum(plan) / len(plan)) if plan else None
        lines.append(
            f"| `{model}` "
            f"| {_rate([not r.get('selection_parse_failed') for r in recs])} "
            f"| {_stat([r.get('selected_count') for r in recs], 'mean')} "
            f"| {_stat([r.get('selection_prereq_added_count') for r in recs], 'mean')} "
            f"| {sel_mean:.0%} "
            f"| {f'{plan_mean:.0%}' if plan_mean is not None else '—'} "
            f"| {f'{(sel_mean - plan_mean):.0%}' if plan_mean is not None else '—'} |"
        )
    return lines + [""]


def _scenario_breakdown(records: list[dict], mode: str) -> list[str]:
    recs = [r for r in records if r.get("stage") == mode and not r.get("error")]
    if not recs:
        return []
    scenarios = sorted({r.get("scenario") for r in recs if r.get("scenario")})
    models = sorted({r["model"] for r in recs})
    if not scenarios:
        return []
    lines = ["**PLAN_VIABLE by scenario** — a model that only fails the hardest scenario is a "
             "different proposition from one that fails everywhere.", "",
             "| Model | " + " | ".join(scenarios) + " |",
             "|---" * (len(scenarios) + 1) + "|"]
    for model in models:
        cells = [
            _rate([bool(r.get("plan_viable")) for r in recs
                   if r["model"] == model and r.get("scenario") == scenario])
            for scenario in scenarios
        ]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return lines + [""]


def _mode_label(stage: str) -> str:
    return "Mode " + stage.removeprefix("plan_mode_").upper()


def _pp(new: float | None, old: float | None) -> str:
    """Signed percentage-point delta, or an em dash when either side is missing."""
    if new is None or old is None:
        return "—"
    return f"{(new - old) * 100:+.0f}pp"


def _mean(values: list[Any]) -> float | None:
    clean = _numbers(values)
    return sum(clean) / len(clean) if clean else None


def _latency_section(records: list[dict]) -> list[str]:
    """What a student actually waits for.

    TTFT and total are NOT interchangeable here, and reporting one number for both kinds of
    call would misdescribe the product. The QA and explain call sites stream prose straight
    to the browser, so TTFT is the perceived wait and the total is just how long the answer
    kept growing. The plan modes return a JSON object under a grammar constraint — half a plan
    is not a plan, nothing can be rendered until the object closes, so the perceived wait is
    the total and TTFT only measures prompt processing.

    All figures exclude model load: the warmup generations run before anything is measured and
    are discarded. A student hitting a cold server waits for the weights on top of this.
    """
    stages = [s for s in ("qa", "explain") if _group(records, s)]
    plan_stages = [s for s in ("plan_mode_a", "plan_mode_b")
                   if _group(records, s)]
    if not stages and not plan_stages:
        return []

    lines = ["### What a student waits for", "",
             "Wall-clock seconds, measured after warmup so model load is excluded. **TTFT** = "
             "request sent → first content token.", ""]

    if stages:
        lines += [
            "*Streaming call sites* — the student sees prose appear at TTFT, so this is the "
            "perceived wait.", "",
            "| Model | Stage | TTFT p50 | TTFT mean | TTFT p95 | Total p50 | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for model in sorted({r["model"] for r in records if r.get("stage") in stages}):
            for stage in stages:
                recs = [r for r in records if r.get("stage") == stage
                        and r["model"] == model and not r.get("error")]
                if not recs:
                    continue
                ttft = [r.get("ttft_s") for r in recs]
                lines.append(
                    f"| `{model}` | {stage} | {_stat(ttft, 'p50', 2)} "
                    f"| {_stat(ttft, 'mean', 2)} | {_stat(ttft, 'p95', 2)} "
                    f"| {_stat([r.get('total_s') for r in recs], 'p50')} | {len(recs)} |"
                )
        lines.append("")

    if plan_stages:
        lines += [
            "*Structured plan call sites* — grammar-constrained JSON. Nothing is renderable "
            "until the object closes, so **total** is the wait and TTFT is prompt processing "
            "only. Mode A's TTFT is its first request's; under `--mitigate` its total covers "
            "every retry, which is also what the student would sit through.", "",
            "| Model | " + " | ".join(
                f"{_mode_label(s)} TTFT p50 | {_mode_label(s)} total p50 | "
                f"{_mode_label(s)} total p95"
                for s in plan_stages
            ) + " |",
            "|---" * (3 * len(plan_stages) + 1) + "|",
        ]
        for model in sorted({r["model"] for r in records if r.get("stage") in plan_stages}):
            cells = []
            for stage in plan_stages:
                recs = [r for r in records if r.get("stage") == stage
                        and r["model"] == model and not r.get("error")]
                cells += [
                    _stat([r.get("ttft_s") for r in recs], "p50", 2),
                    _stat([r.get("total_s") for r in recs], "p50"),
                    _stat([r.get("total_s") for r in recs], "p95"),
                ]
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
        lines.append("")

    return lines


def _unsatisfiable_section(records: list[dict]) -> list[str]:
    """The scenario where the honest answer is "this doesn't fit".

    Excluded from the headline PLAN_VIABLE because coverage cannot reach 1.0 for anyone. What
    it tests instead is what a model does when it cannot succeed: report the shortfall, or
    fabricate room for it. Inventing an extra semester or blowing through the credit cap to
    make the numbers work is the single most damaging behaviour on a real advising site,
    because it looks like a complete plan.

    """
    stages = [
        s for s in ("plan_mode_b",)
        if any(r.get("stage") == s and r.get("expect_unsatisfiable") and not r.get("error")
               for r in records)
    ]
    if not stages:
        return []
    lines = ["### When nothing fits — the honesty check", "",
             "One scenario in the fixture is unsatisfiable by design (one semester left, more "
             "requirements than can fit). Nobody scores PLAN_VIABLE here. The question is "
             "whether the model says so.", "",
             "| Model | Mode | Stayed in horizon | Respected credit cap | Declared what didn't fit | Violations/plan |",
             "|---|---|---|---|---|---|"]
    for stage in stages:
        recs = [r for r in records if r.get("stage") == stage
                and r.get("expect_unsatisfiable") and not r.get("error")]
        for model in sorted({r["model"] for r in recs}):
            mine = [r for r in recs if r["model"] == model]
            caps = [not (r.get("violation_counts") or {}).get("credit_cap_violation")
                    for r in mine]
            horizon = [not r.get("over_horizon") for r in mine]
            declared = [bool(r.get("declared_unplanned")) for r in mine]
            vio = [len(r.get("violations") or []) for r in mine]
            lines.append(
                f"| `{model}` | {_mode_label(stage)} | {_rate(horizon)} | {_rate(caps)} "
                f"| {_rate(declared)} | {(sum(vio) / len(vio)):.1f} |"
            )
    return lines + [""]


def _qa_section(records: list[dict]) -> list[str]:
    groups = _group(records, "qa")
    if not groups:
        return []
    lines = ["### Grounded QA (the RAG summarization step)", "",
             "Retrieval is fixed and identical for every model — the chunks come from "
             "`questions.yaml`, not a live pgvector query — so this measures the chat model "
             "only, not the embedding model. `Faith-flagged` and `Recall-flagged` are "
             "HEURISTICS: entity-level triage, not verdicts. Grade the review queue before "
             "quoting a faithfulness number.", "",
             "**Read the two behaviour columns separately.** `Answered OK` is over the items "
             "whose context supports an answer; `Abstained OK` is over the items where it does "
             "not and declining IS the correct output. A model that never abstains scores well "
             "on the first and zero on the second, and the average of the two hides exactly "
             "the failure that matters for an advising bot.", "",
             "| Model | Answered OK | Abstained OK | Auto-pass | Faith-flagged | "
             "Recall-flagged | Median s |",
             "|---|---|---|---|---|---|---|"]
    for model, recs in groups.items():
        answer_recs = [r for r in recs if r.get("expected_behavior") == "answer"]
        abstain_recs = [r for r in recs if r.get("expected_behavior") == "abstain"]
        lines.append(
            f"| `{model}` "
            f"| {_rate([bool(r.get('behavior_ok')) for r in answer_recs])} "
            f"| {_rate([bool(r.get('behavior_ok')) for r in abstain_recs])} "
            f"| {_rate([bool(r.get('qa_auto_pass')) for r in recs])} "
            f"| {_rate([bool(r.get('faithfulness_flags')) for r in recs])} "
            f"| {_rate([bool(r.get('recall_flags')) for r in recs])} "
            f"| {_median([r.get('total_s') for r in recs])} |"
        )
    lines.append("")

    # BY TOPIC. `questions.yaml` carries `topic` (course | policy | process) because a bot can
    # be fluent about prerequisites and useless about how to CODO — those questions retrieve
    # different corpora and fail differently, and a single QA percentage averages the two into
    # a number that describes neither.
    topics = sorted({r.get("topic") for r in records
                     if r.get("stage") == "qa" and r.get("topic")})
    if len(topics) > 1:
        lines += ["**Behaviour by topic** (share of items handled correctly, answer and "
                  "abstain pooled — the split above is the one to read for safety):", "",
                  "| Model | " + " | ".join(f"`{t}`" for t in topics) + " |",
                  "|---|" + "---|" * len(topics)]
        for model, recs in groups.items():
            cells = [_rate([bool(r.get("behavior_ok")) for r in recs if r.get("topic") == t])
                     for t in topics]
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _explain_section(records: list[dict]) -> list[str]:
    groups = _group(records, "explain")
    if not groups:
        return []
    lines = ["### Explain-plan", "",
             "Every model explains the SAME deterministic baseline plan, which isolates "
             "explanation quality from planning quality.", "",
             "`Contradicts plan` counts explanations making a claim the plan itself refutes — "
             "a course placed in the wrong semester, an ordering the plan reverses, or a plan "
             "described as complete while the planner is reporting courses it could not fit "
             "(see `scorers.explain_flags`). Unlike the faithfulness heuristic next to it, "
             "these are checked against structured data the harness generated, so a flag here "
             "is a real disagreement rather than triage. `Auto-pass` requires both clean.", "",
             "| Model | Auto-pass | Contradicts plan | Faith-flagged | Truncated | "
             "Median output tokens | Median s |",
             "|---|---|---|---|---|---|---|"]
    for model, recs in groups.items():
        lines.append(
            f"| `{model}` "
            f"| {_rate([bool(r.get('explain_auto_pass')) for r in recs])} "
            f"| {_rate([bool(r.get('explain_flags')) for r in recs])} "
            f"| {_rate([bool(r.get('faithfulness_flags')) for r in recs])} "
            f"| {_rate([bool(r.get('truncated')) for r in recs])} "
            f"| {_median([r.get('eval_count') for r in recs])} "
            f"| {_median([r.get('total_s') for r in recs])} |"
        )
    lines.append("")

    # WHAT THE FLAGS ACTUALLY SAID. A percentage tells you an explanation contradicted the
    # plan; only the text tells you whether it moved a course or claimed a short plan was
    # finished, and those are different severities.
    flagged = [(r.get("model"), flag) for r in records if r.get("stage") == "explain"
               for flag in (r.get("explain_flags") or [])]
    if flagged:
        lines += ["Every contradiction found, verbatim:", ""]
        lines += [f"- `{model}` — {flag}" for model, flag in flagged[:20]]
        if len(flagged) > 20:
            lines.append(f"- _...and {len(flagged) - 20} more; see the review queue._")
        lines.append("")
    return lines


def _env_section(records: list[dict]) -> list[str]:
    envs = [r for r in records if r.get("stage") == "env"]
    if not envs:
        return []
    lines = ["### Environment (measured, not assumed)", "",
             "| Model | VRAM delta MB | GPU offload | Server n_ctx | Matches config? |",
             "|---|---|---|---|---|"]
    for env in envs:
        offload = env.get("offload") or {}
        layers = offload.get("layers") or "UNVERIFIED"
        match = env.get("ctx_matches_config")
        lines.append(
            f"| `{env['model']}` | {env.get('vram_delta_mb')} "
            f"| {layers} ({offload.get('source', '?')}) | {env.get('server_n_ctx')} "
            f"| {'yes' if match else ('**NO**' if match is False else '?')} |"
        )
    return lines + [""]


def _errors_section(records: list[dict]) -> list[str]:
    errors = [r for r in records if r.get("error")]
    if not errors:
        return []
    lines = ["### Failures", "",
             "A model that would not load, OOMed, or timed out is a finding, not a gap. Listed "
             "here rather than silently omitted from the tables above.", ""]
    seen = set()
    for rec in errors:
        key = (rec.get("model"), str(rec.get("error"))[:120])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{rec.get('model')}` ({rec.get('stage')}): {str(rec.get('error'))[:300]}")
    return lines + [""]


def _guards(records: list[dict], cfg: dict, fixture_meta: dict, database_meta: dict) -> list[str]:
    lines: list[str] = []
    ok = True


    names = [m["name"] for m in cfg.get("models", [])]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        ok = False
        lines.append(f"- ⚠️ **duplicate model entries in config.yaml**: {', '.join(dupes)} — a "
                     "model listed twice runs twice and pools both conditions into one row.")

    for stage in ("plan_mode_a", "plan_mode_b", "qa", "explain"):
        hashes = {r.get("static_hash") for r in records
                  if r.get("stage") == stage and r.get("static_hash")}
        if len(hashes) > 1:
            ok = False
            lines.append(f"- ⚠️ **mixed static-prompt hashes in `{stage}`** ({sorted(hashes)}) — "
                         "these records came from different prompts and MUST NOT be compared. "
                         "Re-run, or split the file by hash.")

    fixture_hashes = {r.get("fixture_hash") for r in records if r.get("fixture_hash")}
    if len(fixture_hashes) > 1:
        ok = False
        lines.append(f"- ⚠️ **mixed fixture hashes** ({sorted(fixture_hashes)}) — the scoring "
                     "authority (the program's own database export) changed partway through.")

    if fixture_meta and not fixture_meta.get("verified", False):
        ok = False
        lines.append("- ⚠️ **plan fixture is `verified: false`** — its prerequisite edges and "
                     "term offerings are hand-written, because Purdue's catalog publishes "
                     "neither. Model *rankings* are usable (every model is scored against the "
                     "same rules); any absolute claim about real degree progress is not, until "
                     "`python run.py fixture-check` has been run against a populated catalog DB.")

    mismatched = [r["model"] for r in records
                  if r.get("stage") == "env" and r.get("ctx_matches_config") is False]
    if mismatched:
        ok = False
        lines.append(f"- ⚠️ **context-size mismatch**: {', '.join(mismatched)} ran at a context "
                     "size the config did not ask for and are not comparable to the rest.")

    # Mode B is scored against `ctx.database` (see `real_scoring.py`), so a program mismatch no
    # longer corrupts its numbers the way it used to — they honestly describe whichever program
    # was actually shown. What this guard still catches: `real_db.program_id`/`--major` naming
    # a DIFFERENT program than the one the fixture's scenarios were written to describe (student
    # names, "Already completed" lists, credit targets — Mode A/C context). A low overlap means
    # every Mode B number in this report is honest about a program nobody meant to test. See
    # `runner.write_meta`'s `fixture_overlap`.
    overlap = database_meta.get("fixture_overlap")
    if overlap is not None and overlap < 0.5:
        ok = False
        program_name = database_meta.get("program_name") or "the configured program"
        lines.append(
            f"- ⚠️ **Mode B was shown a different program than intended**: "
            f"`{program_name}`'s catalog overlaps only {overlap:.0%} with the plan fixture's "
            f"course universe, and the fixture's own scenarios (student names, completed "
            f"courses, credit targets) were written for a different program. Mode B's numbers "
            f"are honest about `{program_name}` — they are just probably not the program you "
            f"meant to point `real_db.program_id`/`--major` at."
        )

    n = len(fixture_meta.get("scenarios") or [])
    floor = cfg.get("decision", {}).get("min_plan_scenarios", 8)
    if 0 < n < floor:
        ok = False
        ci = int(1.96 * (0.5 / (n ** 0.5)) * 100)
        lines.append(
            f"- ⚠️ **statistical power**: {n} scenarios is the real sample size — the replicates "
            f"are correlated repeats of the same item, not independent samples. A binomial 95% "
            f"CI at n={n} is roughly ±{ci}pp, the same magnitude as any threshold worth setting. "
            f"If two models differ by less than that, the correct response is *write more "
            f"scenarios*, not *pick a winner*. Target: {floor}."
        )

    if ok:
        lines.append("- All guards passed.")
    return lines + [""]


def _review_queue(records: list[dict], out_path: Path) -> int:
    """Every free-text answer plus every non-viable plan, for manual grading.

    Each row carries ``reason`` — what a grader is being asked to decide. Without it the file
    is an undifferentiated pile and the expensive judgements (does this answer contradict the
    context? is this prereq edge real?) are indistinguishable from filler.
    """
    rows = []
    for rec in records:
        if rec.get("needs_review"):
            flags = rec.get("faithfulness_flags") or []
            if rec.get("behavior_mixed"):
                # Refused AND delivered grounded content — behavior_ok cannot adjudicate it.
                reason = "mixed: refused but also answered — did it answer the question asked?"
            elif flags:
                reason = "flagged: unsupported entity in the answer"
            else:
                reason = "routine faithfulness check"
            rows.append({
                "model": rec.get("model"), "stage": rec.get("stage"),
                "item": rec.get("question_id") or rec.get("scenario"),
                "run_idx": rec.get("run_idx"),
                "reason": reason,
                "expected_behavior": rec.get("expected_behavior"),
                "behavior_mixed": rec.get("behavior_mixed"),
                "faithfulness_flags": flags,
                "recall_flags": rec.get("recall_flags"),
                "output": rec.get("output"),
                "grade": None, "notes": None,
            })
        elif (rec.get("stage") in ("plan_mode_a", "plan_mode_b")
              and rec.get("plan_viable") is False):
            violations = rec.get("violations") or []
            # A plan for the deliberately-unsatisfiable scenario is non-viable BY DESIGN. With
            # no violations to confirm there is nothing for a human to decide, and queueing it
            # anyway buried the real items: every Mode A "failure" in the 2026-07-25 run was
            # this case, 12 rows of nothing.
            if rec.get("expect_unsatisfiable") and not violations:
                continue
            rows.append({
                "model": rec.get("model"), "stage": rec.get("stage"),
                "item": rec.get("scenario"), "run_idx": rec.get("run_idx"),
                "reason": ("confirm each violation is real and not a wrong fixture edge"
                           if violations else "no violations — confirm the coverage gap is real"),
                "expect_unsatisfiable": bool(rec.get("expect_unsatisfiable")),
                "violations": violations,
                "missing_requirements": rec.get("missing_requirements"),
                "semesters": rec.get("semesters"),
                "grade": None, "notes": None,
            })
    out_path.write_text("\n".join(json.dumps(r, default=str) for r in rows))
    return len(rows)


def spec_sweep_table(root: Path) -> list[str]:
    """Markdown decode-throughput table across the `run.py spec-sweep` arms, or [] if none ran.

    HERE, NOT IN run.py, so `report` and `spec-sweep --compare-only` render one table from one
    implementation — report.py already owns every other results-to-markdown table, and a second
    copy in the CLI is how the two drift into disagreeing about the same numbers.

    Reads `results/results_spec_<type>/runs_baseline.jsonl`, one directory per arm, written by
    `run.py spec-sweep`. Speculative decoding changes only HOW tokens are produced, never which
    ones, so the plan/QA tables above are the arms' quality check and this one is purely about
    speed — the two belong in the same report rather than in separate places.
    """
    results = root / "results"
    if not results.is_dir():
        return []
    arms = [(d.name[len("results_spec_"):].replace("_", "-"), d)
            for d in sorted(results.iterdir())
            if d.is_dir() and d.name.startswith("results_spec_")]
    if not arms:
        return []
    # "none" is the baseline every other arm is divided by, so it has to come first whatever
    # the directory sort produced.
    arms.sort(key=lambda a: (a[0] != "none", a[0]))

    def decode_tps(recs: list[dict[str, Any]]) -> float | None:
        vals = [r["eval_count"] / (r["predicted_ms"] / 1000) for r in recs
                if (r.get("eval_count") or 0) > 20 and (r.get("predicted_ms") or 0) > 0]
        return statistics.median(vals) if vals else None

    loaded = [(name, _load(d / "runs_baseline.jsonl")) for name, d in arms]
    by_arm = [(name, _by_model(recs)) for name, recs in loaded]
    old = _by_model(_load(root / "results_old/results/runs_baseline.jsonl"))
    base_name, base_by_model = by_arm[0]

    header = ["Model"] + [f"`{n}` tok/s" for n, _ in by_arm]
    if len(by_arm) > 1:
        header += ["Change", "Identical output"]
    header += ["results_old"]
    lines = [
        "## Speculative decoding — `run.py spec-sweep`",
        "",
        "Median decode throughput per model, one column per `--spec-type` arm. Speculative "
        "decoding drafts cheap tokens and verifies them in one batched pass: it trades spare "
        "compute for fewer weight reads. Rejected drafts are wasted compute, so an arm CAN "
        "come out slower than `none` — that is a result, not a bug.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for model in sorted({m for _, bm in by_arm for m in bm} | set(old)):
        base = decode_tps(base_by_model.get(model, []))
        row = [f"`{model}`"]
        for _, bm in by_arm:
            tps = decode_tps(bm.get(model, []))
            row.append(f"{tps:.1f}" if tps else "—")
        if len(by_arm) > 1:
            alt = decode_tps(by_arm[1][1].get(model, []))
            row.append(f"{alt / base:.2f}x" if (base and alt) else "—")
            # A throughput win on an arm whose TEXT diverged is not comparing like with like:
            # at a fixed seed the arms should produce nearly the same tokens.
            def key(r: dict[str, Any]) -> tuple:
                return (r.get("stage"), r.get("question_id"), r.get("run_idx"))
            a = {key(r): r.get("output") for r in base_by_model.get(model, [])}
            b = {key(r): r.get("output") for r in by_arm[1][1].get(model, [])}
            shared = [k for k in a if k in b]
            row.append(f"{sum(1 for k in shared if a[k] == b[k])}/{len(shared)}"
                       if shared else "—")
        o = decode_tps(old.get(model, []))
        row.append(f"{o:.1f}" if o else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"Baseline column is `{base_name}`. `results_old` was recorded on an older llama.cpp "
        "build (this box was bumped to b10362 for Muse Glimmer support), so it is a drift "
        "cross-check — not the baseline the speculative-decoding comparison rests on.",
        "",
    ]
    return lines


def _by_model(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:
        if rec.get("model"):
            out.setdefault(rec["model"], []).append(rec)
    return out


def generate_report(root: Path) -> Path:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    # Same override every other entry point honours (`runner.load_context`), so
    # `MODEL_EVAL_RESULTS_DIR=... run.py report` reads the directory it just wrote instead of
    # config.yaml's default.
    results = root / os.environ.get("MODEL_EVAL_RESULTS_DIR", cfg["paths"]["results_dir"])
    results.mkdir(exist_ok=True)
    baseline = _load(results / "runs_baseline.jsonl")
    mitigated = _load(results / "runs_mitigated.jsonl")
    thinking = _load(results / "runs_thinking.jsonl")
    meta_path = results / "meta_baseline.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fixture_meta = meta.get("fixture", {})
    run_meta = meta.get("run", {})
    # ONE PER SCENARIO now (`runner.write_meta`'s `database_by_scenario`) — each scenario's own
    # gen_ed_preference/world_language narrows the real database differently. The guard below
    # only cares whether SOME scenario was shown the wrong program, so the worst (lowest
    # overlap) one is the representative sample, not an arbitrary first entry.
    database_by_scenario = meta.get("database_by_scenario", {})
    database_meta = (
        min(database_by_scenario.values(),
            key=lambda d: d.get("fixture_overlap") if d.get("fixture_overlap") is not None else 1.0)
        if database_by_scenario else meta.get("database", {})
    )

    lines = [
        "# Model evaluation — BoilerAdvisor",
        "",
        f"Fixture `{fixture_meta.get('path', '?')}` (hash `{fixture_meta.get('hash', '?')}`, "
        f"{fixture_meta.get('courses', '?')} courses, "
        f"{fixture_meta.get('requirement_groups', '?')} requirement groups)  ",
        f"Run {meta.get('timestamp', '?')} · tasks {meta.get('tasks', '?')} · "
        f"num_ctx {run_meta.get('num_ctx', '?')} · temp {run_meta.get('temperature', '?')} · "
        f"seed {run_meta.get('seed', '?')} · "
        f"{run_meta.get('runs_per_pair', '?')} replicates · "
        f"{run_meta.get('parallel', 1)} server slot(s)",
        "",
        "## 1. Plan of study — the feature this is really about",
        "",
        "A plan is VIABLE only if it has zero hard violations (prerequisite ordering, term "
        "offerings, credit cap, hallucinated course codes, duplicates) **and** covers every "
        "degree requirement. One prerequisite mistake anywhere in eight semesters fails the "
        "whole plan. That is the correct standard: a student following it gets turned away at "
        "registration.",
        "",
    ]

    lines += _plan_section(
        baseline, "plan_mode_b", "Mode B — the model builds the whole schedule",
        "No production call site does this today. It is the discriminating measurement: the "
        "model gets the catalog and the requirements and must produce the schedule itself. "
        "Read it as *what would we gain, or risk, by trusting the model with sequencing?*",
    )
    lines += _scenario_breakdown(baseline, "plan_mode_b")
    if thinking:
        thinking_budget = (cfg.get("thinking") or {}).get("budget_tokens", "?")
        lines += _plan_section(
            thinking, "plan_mode_b_thinking",
            "Mode B — thinking variant (`--tasks plan_b_thinking`)",
            "Same prompt, same schema, same scorer as Mode B above — the only difference is "
            f"reasoning ON and budget-capped at {thinking_budget} tokens before the plan, "
            "launched on its own server (see config.yaml's `thinking:` block). Compare this "
            "table's PLAN_VIABLE/coverage directly against Mode B's own table above — that "
            "delta is the answer to \"does pre-plan reasoning change anything.\" Not the "
            "removed post-hoc `rationale` field: that was written after the plan was already "
            "fixed and could never have changed it.",
        )
        avg_chars = _mean([r.get("reasoning_chars") for r in thinking
                           if r.get("stage") == "plan_mode_b_thinking"])
        if avg_chars:
            lines += [f"Average reasoning length: {avg_chars:.0f} characters. Full reasoning "
                      "text is in each run's transcript (`transcripts/<model>/"
                      "plan_mode_b_thinking/`) under the `## Reasoning` heading. A model "
                      "whose average is 0 has no reasoning channel its template will open — "
                      "its rows here ARE Mode B, and a flat score against Mode B means "
                      "nothing happened, not that thinking failed to help.", ""]
    lines += _plan_section(
        baseline, "plan_mode_a", "Mode A — the app's real revise-plan path",
        "This is `advisor_agent.revise_plan` as shipped: the model emits a `PlanEditProposal` "
        "and the deterministic planner rebuilds the schedule. Viability here should be ~100% "
        "for every model **by construction** — if it isn't, the harness's vendored planner has "
        "drifted from the app's, and that is itself the finding.",
    )
    lines += _mode_a_extra(baseline)
    lines += _selection_section(baseline)
    lines += _unsatisfiable_section(baseline)
    lines += _qa_section(baseline)
    lines += _explain_section(baseline)
    lines += _latency_section(baseline)

    section = 2
    if mitigated:
        lines += [f"## {section}. After mitigation", "",
                  "Mitigation = the multi-iteration revise loop (`revise_max_iterations`) plus "
                  "retry-on-unusable-proposal: the fixes that cost $0. A gap that closes here "
                  "never justified hardware.", ""]
        lines += _plan_section(mitigated, "plan_mode_b", "Mode B (mitigated)", "")
        lines += _plan_section(mitigated, "plan_mode_a", "Mode A (mitigated)", "")
        lines += _mode_a_extra(mitigated)
        section += 1

    lines += [f"## {section}. Mode C — what the repair passes bought", ""]
    lines += mode_c_lines(results)
    section += 1

    lines += [f"## {section}. Environment and failures", ""]
    lines += _env_section(baseline)
    lines += _errors_section(baseline)

    section += 1
    lines += [f"## {section}. Validity guards", ""]
    lines += _guards(baseline + mitigated, cfg, fixture_meta, database_meta)

    queue_path = results / "review_queue.jsonl"
    count = _review_queue(baseline + mitigated, queue_path)
    section += 1
    lines += [
        f"## {section}. Your turn",
        "",
        f"`{queue_path.name}` holds {count} items awaiting manual grading: every free-text "
        "answer (faithfulness is not machine-checkable) and every non-viable plan (so you can "
        "confirm the violation the harness found is real and not a fixture bug — the fastest "
        "way to discover a wrong prereq edge is to see three good models all 'fail' the same "
        "course). The plan-viability numbers above stand on their own; the faithfulness "
        "numbers do not until this queue is graded.",
        "",
    ]

    lines += spec_sweep_table(root)

    report_path = results / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path} ({count} items in the review queue)")
    return report_path


# --- cross-school summary (multi-school `run.py run`) ---------------------------------------


def _discover_school_dirs(root: Path) -> list[tuple[str, Path]]:
    """(slug, dir) for every `results/results_<slug>/` the multi-school loop wrote — see
    `run.py`'s `_run_all_schools`. Sorted by slug for stable output."""
    base = root / "results"
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        # `results_spec_<type>/` are spec-sweep ARMS, not schools — same school for every one
        # of them, differing only by --spec-type. Pooling them here would double-count the one
        # school they ran and invent slugs like "spec_none" in the per-school grid. They are
        # reported by `spec_sweep_table` instead.
        if d.is_dir() and d.name.startswith("results_") \
                and not d.name.startswith("results_spec_"):
            out.append((d.name[len("results_"):], d))
    return out


def generate_summary(root: Path) -> Path:
    """`results/summary.md` — one file answering "which model, across every school" instead of
    per-school `report.md`'s "which model, for this one school". Two views:

      1. OVERALL: every school's `runs_baseline.jsonl` records pooled into one corpus and run
         through the SAME `_plan_section` table `report.md` uses — a model that only looks
         good on CS/ME (the two schools with the most authoring care) should not look good
         here if it falls apart on the other fourteen.
      2. PER SCHOOL: a compact PLAN_VIABLE/coverage grid, school x model, so a specific bad row
         (one school, one model) is visible even though the overall table averages it away.

    BOTH VIEWS INCLUDE THE THINKING ARM (`runs_thinking.jsonl`) when a school has one. It is
    kept as its own corpus rather than pooled into the baseline — the two were generated
    against servers launched with opposite reasoning settings and must never be averaged
    together — but it is REPORTED HERE, in the same file, because a reader asking "did
    thinking help" should not have to know which file each arm lives in.

    Schools that never ran (no `results/results_<slug>/runs_baseline.jsonl`) are silently
    absent from both views — this reports what exists, it does not claim the sweep was
    complete. Cross-reference against the program's own database export if you need to know what's
    missing.
    """
    school_dirs = _discover_school_dirs(root)
    if not school_dirs:
        raise SystemExit(
            "no results/results_<slug>/ directories found — run `python run.py run` (no "
            "--major) first, or pass --major for a single-school `python run.py report`."
        )

    pooled: list[dict[str, Any]] = []
    per_school: dict[str, list[dict[str, Any]]] = {}
    # THE THINKING ARM IS POOLED TOO, in its own corpus rather than mixed into `pooled`.
    # It lives in a separate file per school because it was generated against a separate
    # server (reasoning ON), and that separation is the whole reason its numbers mean
    # anything — but "separate file" was being read as "separate report", so a full sweep
    # produced a cross-school summary that silently omitted an entire arm. One
    # `run.py report --all` now shows both.
    pooled_thinking: list[dict[str, Any]] = []
    per_school_thinking: dict[str, list[dict[str, Any]]] = {}
    fixture_names: dict[str, str] = {}
    for slug, d in school_dirs:
        thinking_recs = _load(d / "runs_thinking.jsonl")
        if thinking_recs:
            per_school_thinking[slug] = thinking_recs
            pooled_thinking.extend(thinking_recs)
        recs = _load(d / "runs_baseline.jsonl")
        if not recs:
            continue
        for rec in recs:
            rec = dict(rec)
            rec["_school"] = slug
        per_school[slug] = recs
        pooled.extend(recs)
        meta_path = d / "meta_baseline.json"
        # The program's display name comes off the meta itself now — `write_meta` records
        # `fixture.name`/`slug`/`program_id`. It used to be read back out of the fixture YAML
        # `fixture.path` pointed at; there is no such file any more (see `fixtures.py`), and a
        # meta written before this change simply falls back to the directory slug.
        fixture_names[slug] = slug
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            fixture_names[slug] = meta.get("fixture", {}).get("name") or slug

    if not pooled:
        raise SystemExit(
            f"found {len(school_dirs)} results/results_<slug>/ dir(s) but none have a "
            f"runs_baseline.jsonl yet — did `python run.py run` finish any school?"
        )

    lines = [
        "# Cross-school summary — every plan fixture, one report",
        "",
        f"{len(per_school)} school(s) with data: "
        + ", ".join(f"`{slug}`" for slug in sorted(per_school)),
        "",
        "No composite score here either — same discipline as `report.md`. This pools every "
        "school's records into one corpus for the OVERALL tables below, then breaks "
        "PLAN_VIABLE back out by school so one bad school can't hide inside a good average.",
        "",
        "## 1. Overall — every school pooled",
        "",
    ]
    lines += _plan_section(
        pooled, "plan_mode_b", "Mode B — select the courses, then schedule them",
        "Pooled across every school's scenarios. A model whose free-form planning only works "
        "for the one or two most heavily-authored fixtures will show up here as worse than its "
        "own single-school report suggests — that gap IS the finding.",
    )
    if pooled_thinking:
        thinking_budget = (yaml.safe_load((root / "config.yaml").read_text())
                           .get("thinking") or {}).get("budget_tokens", "?")
        lines += _plan_section(
            pooled_thinking, "plan_mode_b_thinking",
            "Mode B — thinking variant, pooled across every school",
            "The same select-then-schedule task as the table above, with reasoning ON and "
            f"capped at {thinking_budget} tokens before the plan starts. Read this table "
            "against Mode B's directly above it: same prompt, same schema, same scorer, one "
            "difference. Credit spread is the column to watch if the question is whether "
            "reasoning produces a more evenly balanced schedule.",
        )
        avg_chars = _mean([r.get("reasoning_chars") for r in pooled_thinking
                           if r.get("stage") == "plan_mode_b_thinking"])
        if avg_chars is not None:
            lines += [f"Average reasoning length: {avg_chars:.0f} characters, pooled. Models "
                      "averaging 0 never opened a reasoning channel — their rows are Mode B "
                      "under another name, and must not be read as evidence about thinking.",
                      ""]

    lines += _plan_section(
        pooled, "plan_mode_a", "Mode A — auditing the schedule the student already has",
        "The student arrives with a draft that has known damage in it (courses removed, one "
        "moved ahead of its prerequisites). PLAN_VIABLE here is whether the CORRECTED schedule "
        "the model returned is legal and complete — see the answer-key table in each school's "
        "own report for whether it found the specific damage.",
    )
    lines += _mode_a_extra(pooled)
    lines += _selection_section(pooled)

    lines += ["## 2. Per school — PLAN_VIABLE and requirement coverage", "", ]
    # (mode, label, corpus) — the thinking arm reads from its own per-school records, since
    # they came out of a different file written by a different server.
    grids = [("plan_mode_a", "Mode A", per_school, pooled),
             ("plan_mode_b", "Mode B", per_school, pooled)]
    if pooled_thinking:
        grids.append(("plan_mode_b_thinking", "Mode B — thinking variant",
                      per_school_thinking, pooled_thinking))
    for mode, label, source, corpus in grids:
        models = sorted({r["model"] for r in corpus if r.get("stage") == mode})
        lines += [f"### {label}", "",
                  "| School | Program | " + " | ".join(
                      f"`{m}` viable / coverage" for m in models) + " |"]
        lines += ["|---|---|" + "---|" * len(models)]
        for slug, recs in sorted(source.items()):
            cells = []
            for model in models:
                model_recs = _satisfiable(
                    [r for r in recs if r.get("stage") == mode and r.get("model") == model
                     and not r.get("error")]
                )
                viable = _by_run(model_recs, "plan_viable")
                coverage = [r["requirement_coverage"] for r in model_recs
                            if r.get("requirement_coverage") is not None]
                cov_cell = f"{(sum(coverage) / len(coverage)):.0%}" if coverage else "—"
                cells.append(f"{_rate_with_range(viable)} / {cov_cell}")
            program = fixture_names.get(slug, slug)
            lines.append(f"| `{slug}` | {program} | " + " | ".join(cells) + " |")
        lines.append("")

    # QA AND EXPLAIN, which this file reported on for the single-school view and never here —
    # so the cross-school summary described a planner and stayed silent about the half of the
    # product that answers questions.
    #
    # THE TWO POOL DIFFERENTLY, and saying so matters more than the tables:
    #
    #   QA      `questions.yaml` is ONE fixed bank, identical for every program. Pooling it
    #           across schools does not broaden coverage — it re-runs the same items once per
    #           school, so the pooled number is the same questions with more replicates. Read
    #           it as a stability check on those items, never as "tested on N programs".
    #   EXPLAIN genuinely varies by school: each program produces its own deterministic plan,
    #           so a model is explaining a different artefact every time. This one IS a
    #           cross-program measurement, and the per-school grid below is where a model that
    #           only explains CS well shows up.
    lines += ["## 3. Question answering and explanation", "",
              "Pooled across every school. Note the asymmetry: the QA bank is one fixed set of "
              "questions repeated per school (more replicates, not more coverage), while "
              "explain runs against each program's own plan and therefore does widen with "
              "every school added.", ""]
    qa_lines = _qa_section(pooled)
    explain_lines = _explain_section(pooled)
    lines += qa_lines or ["_No QA records in any school's results._", ""]
    lines += explain_lines or ["_No explain records in any school's results._", ""]

    if explain_lines:
        explain_models = sorted({r["model"] for r in pooled if r.get("stage") == "explain"})
        lines += ["### Explain — per school", "",
                  "A model explaining a different program's plan every row. `Auto-pass` is "
                  "clean on both checks (no claim the plan contradicts, no unsupported "
                  "entity); the parenthesised count is contradictions found.", "",
                  "| School | Program | " + " | ".join(f"`{m}`" for m in explain_models) + " |",
                  "|---|---|" + "---|" * len(explain_models)]
        for slug, recs in sorted(per_school.items()):
            cells = []
            for model in explain_models:
                model_recs = [r for r in recs if r.get("stage") == "explain"
                              and r.get("model") == model and not r.get("error")]
                if not model_recs:
                    cells.append("—")
                    continue
                contradictions = sum(len(r.get("explain_flags") or []) for r in model_recs)
                cells.append(
                    f"{_rate([bool(r.get('explain_auto_pass')) for r in model_recs])} "
                    f"({contradictions})"
                )
            lines.append(
                f"| `{slug}` | {fixture_names.get(slug, slug)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 4. Mode C — what the repair passes bought, across schools", ""]
    lines += mode_c_cross_school_lines(school_dirs)

    summary_path = root / "results" / "summary.md"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text("\n".join(lines))
    print(f"Wrote {summary_path} ({len(per_school)} school(s), {len(pooled)} pooled records)")
    return summary_path
