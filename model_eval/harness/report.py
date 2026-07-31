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
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


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
        "| Model | PLAN_VIABLE | Structure OK | Req. coverage | Ask honoured | Violations/plan | Idle cr | Heavy CS terms | Sem used | Cr spread | Consistency | Median s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
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
        # SOFT breaches only — the hard ones are already in Violations/plan and in the
        # breakdown below. This column is the term a student would call a bad semester but
        # would still be allowed to register for: exactly at the hard cap, over the preferred
        # one. Per plan, so it is readable next to Violations/plan.
        soft = [r["soft_major_overloads"] for r in recs
                if r.get("soft_major_overloads") is not None]
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
        used_cell = (f"{sum(used) / len(used):.1f}/{sum(avail) / len(avail):.1f}"
                     if used and avail else "—")
        spread_cell = f"{(sum(spread) / len(spread)):.1f}" if spread else "—"
        rows.append((
            (sum(1 for v in flat if v) / len(flat)) if flat else -1.0,
            f"| `{model}` | {_rate_with_range(viable)} | "
            f"{_rate([bool(r.get('structure_ok')) for r in recs])} | {cov_cell} | "
            f"{_rate(asserts)} | {vio_cell} | "
            f"{idle_cell} | {soft_cell} | {used_cell} | {spread_cell} | "
            f"{_consistency(recs, 'scenario', 'plan_viable')} | "
            f"{_median([r.get('total_s') for r in recs])} |",
        ))
    lines += [row for _, row in sorted(rows, key=lambda item: item[0], reverse=True)]

    # WHY plans fail is more actionable than how often.
    lines += ["", "**Where plans break** (violation counts across ALL runs, including the "
              "unsatisfiable scenario — a violation is a violation regardless of whether full "
              "coverage was reachable). `CS overload` is the only column here the registrar "
              "would let a student through: it counts terms holding MORE than the hard "
              "major-course limit, which is a plan the student abandons rather than one they "
              "cannot enrol in:", "",
              "| Model | prereq | coreq | term offering | credit cap | CS overload | hallucinated | duplicate |",
              "|---|---|---|---|---|---|---|---|"]
    for model, recs in groups.items():
        counts: Counter = Counter()
        for rec in recs:
            counts.update(rec.get("violation_counts") or {})
        lines.append(
            f"| `{model}` | {counts.get('prereq_violation', 0)} "
            f"| {counts.get('coreq_violation', 0)} "
            f"| {counts.get('term_offering_violation', 0)} "
            f"| {counts.get('credit_cap_violation', 0)} "
            f"| {counts.get('major_overload_violation', 0)} "
            f"| {counts.get('hallucinated_course', 0)} "
            f"| {counts.get('duplicate_course', 0)} |"
        )
    return lines + [""]


def _mode_a_extra(records: list[dict]) -> list[str]:
    groups = _group(records, "plan_mode_a")
    if not groups:
        return []
    lines = [
        "**Did the model actually help?** Mode A's plan is legal no matter what the model says "
        "— the deterministic planner guarantees that, and a hallucinated proposal degrades to a "
        "no-op. These columns are what separates a model that understood the student from one "
        "that returned an empty proposal.",
        "",
        "| Model | Proposal parsed | Touched anything | Grounded | Ask honoured | Plan not worse |",
        "|---|---|---|---|---|---|",
    ]
    for model, recs in groups.items():
        asserts: list[bool] = []
        for rec in recs:
            asserts.extend((rec.get("assertions_passed") or {}).values())
        lines.append(
            f"| `{model}` | {_rate([not r.get('proposal_parse_failed') for r in recs])} "
            f"| {_rate([bool(r.get('proposal_touched_anything')) for r in recs])} "
            f"| {_rate([bool(r.get('proposal_grounded')) for r in recs])} "
            f"| {_rate(asserts)} "
            f"| {_rate([bool(r.get('plan_not_worse')) for r in recs])} |"
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

    Split by mode, because Mode C has a specific way to fail it: the sample plan describes
    eight semesters and this student has one, so a model that anchors on the template has a
    ready-made reason to overrun the horizon.
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
             "| Model | Behavior OK | Auto-pass | Faith-flagged | Recall-flagged | Median s |",
             "|---|---|---|---|---|---|"]
    for model, recs in groups.items():
        lines.append(
            f"| `{model}` | {_rate_with_range(_by_run(recs, 'behavior_ok'))} "
            f"| {_rate([bool(r.get('qa_auto_pass')) for r in recs])} "
            f"| {_rate([bool(r.get('faithfulness_flags')) for r in recs])} "
            f"| {_rate([bool(r.get('recall_flags')) for r in recs])} "
            f"| {_median([r.get('total_s') for r in recs])} |"
        )
    return lines + [""]


def _explain_section(records: list[dict]) -> list[str]:
    groups = _group(records, "explain")
    if not groups:
        return []
    lines = ["### Explain-plan", "",
             "Every model explains the SAME deterministic baseline plan, which isolates "
             "explanation quality from planning quality. Faith-flagged is a heuristic.", "",
             "| Model | Faith-flagged | Truncated | Median output tokens | Median s |",
             "|---|---|---|---|---|"]
    for model, recs in groups.items():
        lines.append(
            f"| `{model}` | {_rate([bool(r.get('faithfulness_flags')) for r in recs])} "
            f"| {_rate([bool(r.get('truncated')) for r in recs])} "
            f"| {_median([r.get('eval_count') for r in recs])} "
            f"| {_median([r.get('total_s') for r in recs])} |"
        )
    return lines + [""]


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


def _guards(records: list[dict], cfg: dict, fixture_meta: dict) -> list[str]:
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

    sample_hashes = {r.get("sample_plan_hash") for r in records if r.get("sample_plan_hash")}
    if len(sample_hashes) > 1:
        ok = False
        lines.append(f"- ⚠️ **mixed sample-plan hashes** ({sorted(sample_hashes)}) — the "
                     "reference plan Mode C was given changed partway through, so the Mode C − "
                     "Mode B delta is measuring two different reference plans.")

    fixture_hashes = {r.get("fixture_hash") for r in records if r.get("fixture_hash")}
    if len(fixture_hashes) > 1:
        ok = False
        lines.append(f"- ⚠️ **mixed fixture hashes** ({sorted(fixture_hashes)}) — the scoring "
                     "authority (`plan_fixtures/*.yaml`) changed partway through.")

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


def generate_report(root: Path) -> Path:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    results = root / cfg["paths"]["results_dir"]
    results.mkdir(exist_ok=True)
    baseline = _load(results / "runs_baseline.jsonl")
    mitigated = _load(results / "runs_mitigated.jsonl")
    meta_path = results / "meta_baseline.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fixture_meta = meta.get("fixture", {})
    run_meta = meta.get("run", {})

    lines = [
        "# Model evaluation — BoilerAdvisor",
        "",
        f"Fixture `{fixture_meta.get('path', '?')}` (hash `{fixture_meta.get('hash', '?')}`, "
        f"{fixture_meta.get('courses', '?')} courses, "
        f"{fixture_meta.get('requirement_groups', '?')} requirement groups)  ",
        *([f"Sample plan `{meta['sample_plan'].get('path')}` (anchoring diagnostic only) "
           f"(hash `{meta['sample_plan'].get('hash')}`, "
           f"{meta['sample_plan'].get('named_courses')} named courses) — prompt input only, "
           f"it decides no score  "] if meta.get("sample_plan") else []),
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
    lines += _plan_section(
        baseline, "plan_mode_a", "Mode A — the app's real revise-plan path",
        "This is `advisor_agent.revise_plan` as shipped: the model emits a `PlanEditProposal` "
        "and the deterministic planner rebuilds the schedule. Viability here should be ~100% "
        "for every model **by construction** — if it isn't, the harness's vendored planner has "
        "drifted from the app's, and that is itself the finding.",
    )
    lines += _mode_a_extra(baseline)
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

    lines += [f"## {section}. Environment and failures", ""]
    lines += _env_section(baseline)
    lines += _errors_section(baseline)

    section += 1
    lines += [f"## {section}. Validity guards", ""]
    lines += _guards(baseline + mitigated, cfg, fixture_meta)

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

    report_path = results / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path} ({count} items in the review queue)")
    return report_path
