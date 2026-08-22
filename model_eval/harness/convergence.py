"""MODE C — retry-to-convergence.

Modes A and B ask "how good is one attempt?". This asks a different question: **given repeated
attempts, how many does a model need — and how long does it take — to land a plan with no
errors in it?** That is closer to how the product is actually used. A student does not accept
or discard a single plan; they iterate with the assistant until the thing is clean.

THE NAME "MODE C" IS REUSED, DELIBERATELY, AND THERE IS ONE THING TO WATCH.
`results_old8/runs_*.jsonl` on this box already contains records with `"stage": "plan_mode_c"`
carrying a completely different meaning — the removed reference arm (Mode B plus the
department's published sample plan, later Mode B with scorer feedback). Those records are dead
data from a deleted arm, and this mode now owns the name.

What keeps the two apart is not the label, it is everything else: this mode writes
`runs_convergence.jsonl` (the old arm wrote `runs_baseline.jsonl`), its report reads only that
file, and its records carry a `variant` field and a static-prompt hash the old ones do not have.
An analysis that globs `results*/runs_baseline.jsonl` across directories is the one place the
collision could bite — check `variant` is present before pooling anything called `plan_mode_c`.

THREE VARIANTS, TRACKED SEPARATELY, NEVER AVERAGED.

  blind     — same prompt every attempt, new sample (seed moves, temperature optionally
              climbs). No information about what went wrong. Measures the variance of the
              model's own distribution: how often does a good plan fall out by luck?
  feedback  — the validator's specific violations are appended to the next attempt's user
              message. Measures self-correction: can the model read a structured error and
              fix it?
  repair    — the HARNESS deletes the violating placements (deterministic, no model needed)
              and FREEZES everything that validates; the model is only ever asked to fill what
              is still missing. Measures the thing the product would actually ship: how many
              rounds of "fill the gaps" to finish a degree-complete plan.

These are different capabilities and a single "attempts to converge" column that mixed them
would describe neither. They are separate variants in the record and separate columns in the
report.

`repair` IS SCORED ON A DIFFERENT CRITERION, and it has to be — see the block above
`auto_repair`. Its repair step guarantees a clean plan by construction, so "no violations" stops
being evidence about the model; it is scored STRICTLY instead (clean AND every requirement group
covered). Comparing its `attempts p50` against the other two is comparing two different bars,
and the report says so next to the table.

CENSORING IS THE POINT OF THE STATISTICS. A case that runs out of clock at attempt 7 has NOT
told you the model needs 7 attempts — it has told you the model needs *more than* 7. Recording
that as a failure (0%) understates the model and recording it as 7 overstates it. Both are
wrong in ways that move rankings. Every case therefore carries `censored` and `censor_reason`,
and `convergence_report.py` uses a Kaplan-Meier estimator so the censored cases contribute what
they actually know rather than being dropped or flattened.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not touch `plan_scorers.score_plan`. The
convergence criterion is computed FROM that scorer's output (`violation_counts`), so Mode C and
Mode B are validating against one implementation. A second copy of the prereq/credit/term logic
that drifted from the first would be undetectable from the outside and would land squarely in
the convergence numbers.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


from . import plan_scorers
from .fixtures import Fixture, Scenario
from .llamacpp_client import LlamaCppClient, LlamaCppError
from .planner import Profile, generate_plan
from .real_scoring import RealScore

# The spec's definition of a valid plan of study, PREREQUISITES ONLY. Credit-hour caps and term
# offerings were dropped 2026-08-07 — neither is a registration wall a student can't talk their
# way around, and offering data isn't knowable in advance anyway. See `CONVERGENCE_CRITERION_
# CAVEAT` for what this leaves out and why the report prints a second column next to it.
DEFAULT_VIOLATION_CLASSES = (
    "prereq_violation",
)

CONVERGENCE_CRITERION_CAVEAT = """\
`Clean` means the validator had NOTHING left to report — no hard violation, no unmet
requirement, and no semester flagged as overloaded, empty or unusually light. It is strictly
harder than PLAN_VIABLE, which asks only for legality and full coverage and is satisfied by a
plan that hands the student eighteen credits one term and three the next.

AN EMPTY PLAN CANNOT GAME THIS, which was a real hazard under the previous criterion: when
convergence was defined over violations of FILLED slots only, deleting the course named in a
violation always removed it, so the shortest route to "converged" was to schedule nothing.
Requirement coverage is part of the criterion now, and an empty plan fails every group it was
supposed to fill.

`Kept attempt` reports which revision actually reached this table. The loop keeps its BEST
state rather than its last, because a revision pass can make a good plan worse — measured on
qwen3.8-27b, one case improved on attempt 2 and regressed by attempt 3. Read it against
`Improved by feedback`: a mean of 1.0 means every extra pass was wasted tokens."""


# --- confirmed slots (the repair variant's ratchet) -----------------------------------------
#
# STUDENT-AUTHORED LOCKS ARE GONE (2026-08-12). A locked slot used to be a `(semester, course)`
# pair the STUDENT had fixed, hand-authored per program in `locked_slots/<slug>.yaml` and loaded
# here. Those files were the last hand-authored data in the harness and they could not survive
# the move to synthesized students — their keys were per-program scenario ids that no longer
# exist — so they were deleted rather than left to pin nothing while reporting success.
#
# The TYPE stays, because it is also the repair variant's own state: `working_locked` is every
# placement that has survived validation so far, re-shown to the model as "keep these" on the
# next attempt (see `run_case`'s ratchet and `freeze_placements`). Mode C now runs for every
# program, starting from an empty set, with nothing to author first.


# --- the convergence criterion ---------------------------------------------------------------






# --- deterministic repair + the validated-placement ratchet -----------------------------------
#
# WHY THIS VARIANT EXISTS. The `feedback` arm asks the model to delete a course the validator
# has ALREADY identified by name, semester and reason. That is not a reasoning task — it is a
# deletion the harness can do itself, correctly, every time, in microseconds. Worse, it costs a
# whole round trip and the model routinely rewrites the entire plan to do it: gemma4-e4b's first
# case went 17 -> 4 -> 10 -> 14 violations, discarding good placements on every pass, because
# nothing anchored the work it had already got right.
#
# So `repair` splits the job along the line where the split actually is:
#
#   the HARNESS removes violations   — deterministic, decidable, no model needed
#   the MODEL fills what is missing  — the part that needs judgement
#
# and every placement that survives validation is FROZEN into the locked set for the next
# attempt. The frozen set only grows, so good work is never lost and progress is monotonic —
# a ratchet, not a re-roll.
#
# THE CONVERGENCE CRITERION MUST CHANGE WITH IT, and this is not optional. Under the specified
# criterion (zero violations over filled slots, no coverage condition) this variant would
# converge on attempt 1 for EVERY model, every time — the repair step guarantees a clean plan by
# construction, so "clean" stops being evidence about the model at all. `repair` is therefore
# scored on the STRICT criterion: clean AND every requirement group covered. What it measures is
# how many rounds of "fill the gaps" it takes to finish a degree-complete plan.

_V_SEMESTER = re.compile(r"semester (\d+)")
_V_CODE = re.compile(r"^([A-Z]{2,5} \d{3,5})")












def unmet_candidates(fixture: Fixture, profile: Profile,
                     semesters: list[list[str]]) -> set[str]:
    """Courses that would advance an unmet requirement group, and are not already placed."""
    canonical = fixture.canonical
    canon = lambda code: canonical.get(code, code)
    have = {canon(plan_scorers.normalize_code(c)) for s in semesters for c in s}
    have |= {canon(plan_scorers.normalize_code(c)) for c in profile.completed_courses}

    wanted: set[str] = set()
    for group in fixture.requirement_groups:
        courses = group.get("courses", [])
        if group.get("kind") == "all":
            wanted.update(c for c in courses if canon(c) not in have)
        else:
            credits = sum(fixture.credits(c) for c in courses if canon(c) in have)
            if credits < float(group.get("choose_credits") or 0):
                wanted.update(c for c in courses if canon(c) not in have)
    return wanted














def _tokens_per_s(eval_count: int | None, predicted_ms: float | None) -> float | None:
    """Generation throughput from llama.cpp's own timings.

    REPORTED, NEVER FOLDED INTO A QUALITY NUMBER. A model needing 3 attempts at 23 tok/s beats
    a model needing 1 attempt at 6 tok/s on wall-clock and loses on reasoning, and only showing
    both numbers keeps those two facts apart. See the report's throughput column.
    """
    if not eval_count or not predicted_ms:
        return None
    return round(eval_count / (predicted_ms / 1000.0), 2)






# --- the retry loop ---------------------------------------------------------------------------


def run_case(
    ctx,
    client: LlamaCppClient,
    model_cfg: dict,
    scenario: Scenario,
    *,
    variant: str,
    conv_cfg: dict[str, Any],
    out,
) -> dict[str, Any]:
    """MODE C = MODE B WITH THINKING FORCED ON (2026-08-14).

    Identical to Mode B in every other respect — the same deterministic selection, the same
    retrieved context, the same schema, the same scorer, the same record fields — so the pair
    is a controlled comparison and the ONLY difference between a `plan_mode_b` row and a
    `plan_mode_c` row for the same (model, scenario) is whether the model reasoned first.

    "IF THE MODEL HAS A THINKING MODE, IT MUST USE IT." `think=True` is requested per call, and
    it only bites where the template supports it — a model with no reasoning channel simply
    behaves as it does in Mode B, which is the honest outcome rather than a failure. Whether
    reasoning actually happened is recorded per attempt (`reasoning_chars`), so the report can
    separate "thought and did no better" from "never thought at all"; the two look identical in
    a score column and mean opposite things.
    #
    # REQUIRES A REASONING BUDGET AT LAUNCH (`run.reasoning_budget` > 0). `--reasoning off`
    # cannot be undone by a per-request kwarg, so a sweep configured with a 0 budget will run
    # Mode C as a plain second Mode B pass — `run.py check` says so rather than letting it look
    # like a finding.

    WHAT THIS REPLACED: the retry-to-convergence loop, with its attempt budget, plateau
    detection, censoring discipline and Kaplan-Meier estimator. That machinery answered "how
    many attempts does this model need", which was the right question when a single fused call
    had to choose and sequence at once and usually failed. With selection deterministic and the
    course block grouped by requirement, the first attempt is nearly always the last one, and a
    survival curve over a single observation is not a statistic.
    """
    from .runner import _run_staged_plan

    # `run_case`'s signature is fixed by `run_convergence`'s loop; `variant` is retained in the
    # record for continuity with older files but no longer selects behaviour.
    started = time.monotonic()
    _run_staged_plan(
        ctx, client, model_cfg, scenario, 0, out,
        mitigate=False, stage="plan_mode_c", repair_passes=0,
        think_override=True, variant=variant,
    )
    return {"model": model_cfg["name"], "stage": "plan_mode_c", "variant": variant,
            "scenario": scenario.id, "wall_clock_s": round(time.monotonic() - started, 3)}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 3)


# --- preflight ---------------------------------------------------------------------------------


def preflight(ctx, conv_cfg: dict[str, Any]) -> list[str]:
    """Everything checkable before a GPU-minute is spent. Returns the problems found.

    The prerequisite-risk report is the important half. Mode C COMPOUNDS a bad prereq edge:
    the same wrong edge is re-hit on every attempt, so it does not cost a model one violation,
    it costs it the whole run and turns into a fabricated convergence difference.

    It used to also validate the student's locked slots (reachable, not already completed, not
    pinned behind their own prerequisite). Those are gone — see the `LockedSlot` note above —
    so what is left is the reachability check: with no pins at all, a scenario the deterministic
    planner cannot solve is still a scenario no model can, and Mode C would be measuring the
    program's data rather than the model.
    """
    problems: list[str] = []
    for scenario in ctx.fixture.scenarios:
        if scenario.expect_unsatisfiable:
            continue
        plan = generate_plan(scenario.profile, ctx.fixture.catalog)
        score = plan_scorers.score_plan(
            ctx.fixture, scenario.profile, plan_scorers.plan_to_semesters(plan)
        )
        if not score.viable:
            problems.append(
                f"{scenario.id}: the deterministic planner cannot produce a viable plan — "
                f"convergence would be measuring the program's data, not a model."
            )
    return problems


def run_convergence(
    root: Path, *, models: list[str] | None = None, brackets: list[str] | None = None,
    variants: list[str] | None = None, scenarios: list[str] | None = None,
) -> Path:
    """Top level for Mode C. Own results file, own lock, own schema.

    The server lifecycle is deliberately identical to `runner.run_model`'s — same launch flags,
    same warmup-and-discard, same stop between models — because a convergence number taken
    against a differently-configured server is not comparable to anything, including itself.
    """
    from .runner import _env_record, _run_lock, _warmup, load_context
    from .llamacpp_client import LlamaCppClient
    from .server import gpu_memory_used_mb, start_server

    ctx = load_context(root)
    conv_cfg = dict(ctx.cfg.get("convergence") or {})
    if not conv_cfg:
        raise SystemExit("config.yaml has no `convergence:` block.")

    selected = [
        m for m in ctx.cfg["models"]
        if (not models or m["name"] in models)
        and (not brackets or m.get("bracket") in brackets)
    ]
    if not selected:
        raise SystemExit("No models matched the filter.")

    variant_list = list(
        variants or conv_cfg.get("enabled_variants") or ["repair", "feedback", "blind"])
    unknown = [v for v in variant_list if v not in ("blind", "feedback", "repair")]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}")

    cases = [s for s in ctx.fixture.scenarios if not scenarios or s.id in scenarios]
    if not cases:
        raise SystemExit("No scenarios matched the filter.")

    print(prereq_risk_note(ctx.fixture, conv_cfg))
    problems = preflight(ctx, conv_cfg)
    if problems:
        print(f"\n❌ preflight found {len(problems)} problem(s) — no GPU time spent:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\n✅ preflight passed: every locked slot is legal where it is pinned.\n")

    # CROSS-TAG GUARD. `_run_lock` only refuses a second run under the SAME tag, which is
    # enough when the only tags are baseline/mitigated (they are never wanted at once anyway).
    # Mode C breaks that assumption: `.convergence.lock` and `.baseline.lock` are different
    # files, so nothing structural stops `converge` starting on top of a running sweep — and
    # they would fight over the GPU and over port 8099, producing two sets of plausible-looking
    # latency numbers that describe neither run. Convergence is the more fragile of the two,
    # since its whole output is wall-clock.
    for other in sorted(ctx.results_dir.glob(".*.lock")):
        if other.name == ".convergence.lock":
            continue
        try:
            pid = int(other.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise SystemExit(
                f"refusing to start: {other.name} is held by a LIVE process (pid {pid}). "
                f"Mode C measures wall-clock time; sharing the GPU with another run would "
                f"make every number in it meaningless. Wait for that run, or stop it."
            )
        print(f"[converge] ignoring stale lock {other.name} (pid {pid or 'unknown'} is gone)")

    out_path = ctx.results_dir / "runs_convergence.jsonl"
    with _run_lock(ctx.results_dir, "convergence"):
        _write_meta(ctx, conv_cfg, selected, variant_list, cases)
        mode = "a" if out_path.exists() else "w"
        total = len(selected) * len(variant_list) * len(cases)
        print(f"[converge] {len(selected)} model(s) x {len(variant_list)} variant(s) x "
              f"{len(cases)} scenario(s) = {total} case(s) -> {out_path} (mode={mode})")
        print(f"[converge] worst case {total * conv_cfg['timeout_s'] / 3600:.1f} h if every "
              f"case runs to its {conv_cfg['timeout_s']} s timeout")

        with out_path.open(mode) as out:
            for model_cfg in selected:
                name = model_cfg["name"]
                print(f"\n=== {name} ({model_cfg.get('bracket')}) ===")
                vram_baseline = gpu_memory_used_mb()
                try:
                    handle = start_server(ctx.cfg, model_cfg, ctx.log_dir, host=ctx.host)
                except (RuntimeError, TimeoutError) as exc:
                    print(f"[SKIP] {name}: {exc}")
                    out.write(json.dumps(
                        {"model": name, "stage": "error", "error": str(exc)}) + "\n")
                    out.flush()
                    continue
                client = LlamaCppClient(handle.base_url,
                                        timeout_s=ctx.cfg["run"]["request_timeout_s"])
                try:
                    _warmup(ctx, client, model_cfg)
                    env = _env_record(ctx, handle, client, model_cfg, vram_baseline)
                    env["stage"] = "env"
                    out.write(json.dumps(env, default=str) + "\n")
                    out.flush()
                    print(f"[env] {name}: vram_delta={env['vram_delta_mb']} "
                          f"offload={env['offload']} n_ctx={env['server_n_ctx']}")
                    for variant in variant_list:
                        for scenario in cases:
                            rec = run_case(
                                ctx, client, model_cfg, scenario,
                                variant=variant, conv_cfg=conv_cfg, out=out,
                            )
                            # Mode C is one pass with reasoning on — there is no attempt
                            # count to report, so the line says what actually varies.
                            print(f"  [{name}] {scenario.id}: reasoned, "
                                  f"{rec['wall_clock_s']:.0f}s")
                finally:
                    handle.stop(
                        trim_log_lines=ctx.cfg["llamacpp"].get("keep_log_lines", 400))
                    time.sleep(ctx.cfg["llamacpp"].get("cooldown_s", 5))
    return out_path


def _write_meta(ctx, conv_cfg, selected, variants, cases) -> None:
    meta = {
        "tag": "convergence",
        "mode": "D",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": ctx.host,
        "models": [m["name"] for m in selected],
        "variants": list(variants),
        "scenarios": [s.id for s in cases],
        "run": ctx.cfg["run"],
        "llamacpp": ctx.cfg["llamacpp"],
        "convergence": conv_cfg,
        "fixture": {
            "path": ctx.fixture.path.name,
            "hash": ctx.fixture.fixture_hash,
            "verified": ctx.fixture.verified,
        },
        # ONE PER CASE now — each scenario's own gen_ed_preference/world_language narrows the
        # real database differently; see `harness/real_db.py`'s module docstring and
        # `runner.write_meta`'s matching `database_by_scenario`.
        "database_by_scenario": {
            s.id: {"path": (db := ctx.database_for(s)).path.name, "hash": db.db_hash}
            for s in cases
        },
        "static_hashes": {
            "plan_mode_c_by_scenario": {
                s.id: ctx.prompts_for(s).plan_convergence(s, [])[2]
                for s in cases
            },
        },
    }
    (ctx.results_dir / "meta_convergence.json").write_text(json.dumps(meta, indent=2))


def prereq_risk_note(fixture: Fixture, conv_cfg: dict[str, Any]) -> str:
    """The honest state of the prerequisite edges, printed before every Mode C run."""
    edges = sum(len(c.prereqs) for c in fixture.catalog)
    excluded = list(conv_cfg.get("excluded_prereq_edges") or [])
    lines = [
        f"prerequisite edges in the fixture: {edges} across {len(fixture.catalog)} courses",
        f"fixture verified: {fixture.verified}",
    ]
    if not fixture.verified:
        lines.append(
            "  ⚠️  ALL prereq edges are hand-written — Purdue publishes them only in Banner, "
            "whose robots.txt disallows crawling. Mode C compounds a wrong edge across every "
            "attempt of every case, so this is a larger threat here than in Mode A/B."
        )
    lines.append(
        f"excluded edges: {', '.join(excluded) if excluded else 'none'}"
        + ("" if excluded else "  (set convergence.excluded_prereq_edges to exclude known-bad "
                              "edges, e.g. ['CS 38100->MA 26100'])")
    )
    return "\n".join(lines)
