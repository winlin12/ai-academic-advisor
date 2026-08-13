"""Run orchestration.

Measurement discipline enforced here:
  * one llama-server command line, built once from config, applied identically to every model
    (context size, KV type, GPU layers, flash-attn, reasoning-off) — under llama.cpp these are
    LAUNCH flags, so the harness owns the process rather than trusting whoever started one
  * /props is read back after launch and the SERVER's reported context size is recorded, so
    "every model ran at num_ctx" is verified rather than assumed
  * per model: warmup generations (discarded) before anything is measured
  * VRAM = nvidia-smi delta (baseline before launch vs. after warmup)
  * GPU offload parsed from llama-server's own stderr ("offloaded X/Y layers to GPU")
  * the server is stopped between models, so the next model starts from a clean card
  * every record carries the static-prompt hash, the fixture hash, and a config snapshot;
    records whose hashes disagree must never be compared

TASK ORDER PER MODEL. Plan-of-study runs FIRST, before QA and explain. It is the feature the
app will lean on hardest, so if a run has to be cut short (OOM, a model that will not load, a
box that has to go back to being a workstation), the data that survives is the data that
matters. This is deliberate, not incidental.

ONE REQUEST AT A TIME. Every latency here is a single-user latency, which is what makes it
comparable across models and across runs. A concurrent request pool lived here briefly on
2026-07-30 and was removed the next day — it belongs with the deployment question ("what does a
student wait when ten of them are on the site"), not with the model-selection question this
harness answers, and mixing the two in one results file produces medians that describe neither.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import plan_scorers, scorers
from .catalog_export import CatalogDatabase
from .fixtures import Fixture, Scenario
from .llamacpp_client import GenerationResult, LlamaCppClient, LlamaCppError
from .real_db import (
    RealDatabaseBase, build_scenario_database, fetch_real_db_base, fixture_from_database,
    merge_real_db_bases, resolve_poid,
)
from .real_scoring import RealScore, score_against_real_db
from .planner import (
    Plan, Profile, Proposal, apply_proposal, extract_credit_cap, generate_plan,
    set_planning_terms, severity,
)
from .prompts import (
    PROPOSAL_SCHEMA, PromptBuilder, catalog_tags, json_response_format, plan_schema,
    proposal_schema,
)
from .server import (
    ServerHandle, gpu_memory_used_mb, resolve_draft, resolve_host, slot_context,
    speculative_enabled, start_server,
)

# WHAT THE PROMPT COSTS BESIDES THE DATABASE. `build_scenario_database`'s `budget_tokens` sizes
# the rendered database ONLY, but the thing that has to fit in a slot is the whole prompt: the
# schema description, the hard rules, the student's profile and feedback, and the response
# format. Deriving the budget as `slot - max_plan_tokens` therefore over-spends by however much
# that wrapper costs, and single-program scenarios only fit because the trimmer's own
# chars-per-token estimate (2.2) is conservative against the 2.9 this harness measured.
#
# IT STOPPED FITTING when a scenario gained a second major and a minor (2026-08-11): three
# programs' requirement groups rendered to ~14.7k tokens against a 14,336 budget, and
# `check_slot_context` failed the run — correctly. Reserving the wrapper explicitly is the fix,
# and it makes the budget mean "what the database may use" rather than "what the prompt may use
# minus nothing". 1500 is measured against this fixture's Mode B prompt with the database
# removed, rounded up.
_PROMPT_OVERHEAD_TOKENS = 1500

WARMUP_SYSTEM = "You reply with exactly one word."
WARMUP_USER = "Reply with the single word: ready"


@dataclass
class EvalContext:
    cfg: dict[str, Any]
    root: Path
    fixture: Fixture
    # Mode A/QA/Explain's prompt builder. None of the three read `.database`'s content (see
    # `prompts.PromptBuilder`), so this one, built from `real_db_base` with no scenario
    # preference applied, is a safe shared default for them. Mode B/C must NOT use this one —
    # see `database_for`/`prompts_for` below.
    prompts: PromptBuilder
    questions: list[dict[str, Any]]
    host: str
    # The MAXIMAL real-catalog universe, read once from Postgres (see
    # `harness/real_db.fetch_real_db_base`) — every course any scenario's own
    # gen_ed_preference/world_language could select from, prerequisite-closed. Mode B/C never
    # read this directly; they go through `database_for`, which turns it into the one, smaller
    # database a given SCENARIO is actually shown.
    real_db_base: RealDatabaseBase
    budget_tokens: int
    # Per-scenario database/prompt-builder cache — see `database_for`/`prompts_for`. Rebuilding
    # is pure Python (no Postgres round trip), but a scenario's own token-budget fill loop is
    # not free, and every scenario is visited once per model x replicate; memoizing keeps that
    # cost paid once per run instead of once per (model, replicate, scenario) triple.
    _scenario_databases: dict[str, CatalogDatabase] = field(default_factory=dict)
    _scenario_prompts: dict[str, PromptBuilder] = field(default_factory=dict)
    # poid -> that program's own fetched base, for scenarios pursuing a second major or a minor
    # (`Scenario.additional_programs`). Fetched ONCE in `load_context`, like `real_db_base`
    # itself — each one is a full Postgres round trip, and merging is pure Python.
    extra_program_bases: dict[str, RealDatabaseBase] = field(default_factory=dict)

    def database_for(self, scenario: Scenario) -> CatalogDatabase:
        """The real catalog THIS scenario is shown for Mode B/C — `real_db_base` narrowed by
        its own `gen_ed_preference`/`world_language`. Scoring (`score_against_real_db`,
        `real_db.fixture_from_database`) must always be called against this, never against a
        different scenario's database or against `real_db_base` directly — see
        `harness/real_db.py`'s module docstring on why a plan has to be scored against exactly
        what the model was shown.
        """
        cached = self._scenario_databases.get(scenario.id)
        if cached is None:
            # SECOND MAJORS / MINORS merged in first, so the trimmer below sizes the whole
            # multi-program degree against the budget rather than trimming the primary to fit
            # and then blowing past it with the extras.
            base = self.real_db_base
            extras = [
                (self.extra_program_bases[poid], kind, label)
                for poid, kind, label in scenario.additional_programs
                if poid in self.extra_program_bases
            ]
            if extras:
                base = merge_real_db_bases(base, extras)
            cached = build_scenario_database(
                base, gen_ed_preference=scenario.gen_ed_preference,
                world_language=scenario.world_language, budget_tokens=self.budget_tokens,
            )
            self._scenario_databases[scenario.id] = cached
        return cached

    def prompts_for(self, scenario: Scenario) -> PromptBuilder:
        """The `PromptBuilder` wrapping `database_for(scenario)` — what Mode B/C must build
        `plan_freeform`/`plan_convergence` prompts from."""
        cached = self._scenario_prompts.get(scenario.id)
        if cached is None:
            cached = PromptBuilder(self.fixture, self.database_for(scenario))
            self._scenario_prompts[scenario.id] = cached
        return cached

    @property
    def results_dir(self) -> Path:
        d = self.root / self.cfg["paths"]["results_dir"]
        d.mkdir(exist_ok=True)
        return d

    @property
    def log_dir(self) -> Path:
        d = self.results_dir / "server_logs"
        d.mkdir(exist_ok=True)
        return d

    @property
    def transcript_dir(self) -> Path:
        d = self.results_dir / "transcripts"
        d.mkdir(exist_ok=True)
        return d

    def write_transcript(
        self, model: str, stage: str, item: str, run_idx: int,
        system: str, user: str, output: str, verdict: str = "", detail: str = "",
    ) -> None:
        """Save one full exchange as readable markdown.

        The raw text is already in runs_*.jsonl, but a JSONL field is not something you can
        actually read — and reading what the model literally said is how you find out that a
        metric is measuring the wrong thing. (Exactly how the "grounded 0%" result turned out
        to be a model writing whole semester layouts into a field expecting course codes: the
        number alone looked like a scoring bug.) One file per exchange, foldered by model, so
        diffing two models on the same item is a plain `diff`.
        """
        if not self.cfg["run"].get("save_transcripts", True):
            return
        safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
        d = self.transcript_dir / safe(model) / safe(stage)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{safe(item)}__run{run_idx}.md").write_text(
            f"# {model} · {stage} · {item} · run {run_idx}\n\n"
            + (f"**Verdict:** {verdict}\n\n" if verdict else "")
            + (detail + "\n" if detail else "")
            + f"## System prompt\n\n```\n{system}\n```\n\n"
            f"## User prompt\n\n```\n{user}\n```\n\n"
            f"## Model output\n\n```\n{output}\n```\n",
            encoding="utf-8",
        )


def _score_detail(
    score: plan_scorers.PlanScore, semesters: list[list[str]], profile: Profile,
) -> str:
    """Everything the scorer decided, spelled out, above the prompts in the transcript.

    The verdict line carries counts ("violations={'prereq_violation': 3}"), which tells you a
    plan failed but not why — and "why" is where the useful findings have come from. Reading a
    transcript is how STAT 35000 turned out to be a corequisite of CS 37300 rather than a
    prerequisite, which was 15% of every prereq violation in the corpus and was suppressing half
    of one model's Mode B viability. That is only findable if the individual violations, the
    requirements still short, and the per-semester shape are all on the page next to the plan.
    """
    lines: list[str] = ["## Scoring", ""]
    lines.append(
        f"**PLAN_VIABLE: {str(score.viable).upper()}** — {len(score.violations)} violation(s), "
        f"requirement coverage {score.requirement_coverage:.0%}, "
        f"{score.semesters_used}/{score.semesters_available} semesters used, "
        f"credit spread {score.credit_spread}"
    )
    lines.append("")

    if score.violations:
        counts = ", ".join(f"{k} x{v}" for k, v in sorted(score.violation_counts.items()))
        lines += [f"### Violations ({len(score.violations)}) — {counts}", ""]
        lines += [f"- `{v}`" for v in score.violations]
        lines.append("")
    else:
        lines += ["### Violations", "", "None — every hard rule held.", ""]

    if score.missing_requirements:
        lines += ["### Requirements still short", ""]
        lines += [f"- {m}" for m in score.missing_requirements]
        lines.append("")

    if score.semester_credits:
        terms = plan_scorers._terms(profile, len(score.semester_credits))
        # BOTH credit numbers, because they mean different things and the table marks them
        # differently: `asked` is what the student wants (over it is *heavy*, and shows up as a
        # failed `max_credits_at_most` below), `ceiling` is what the registrar allows (over it
        # is **OVER**, a hard violation). Printing only one of them is what made a 17-credit
        # term read as the same kind of failure as a broken prereq chain.
        asked = profile.max_credits_per_semester
        ceiling = profile.hard_credit_cap
        soft = min(profile.preferred_major_courses_per_semester,
                   profile.max_major_courses_per_semester)
        hard = profile.max_major_courses_per_semester
        lines += [
            f"### Semesters (credits: {asked} asked, {ceiling} hard limit; "
            f"{profile.major_subject} cap {hard}, {soft} preferred)", "",
            f"| # | term | credits | {profile.major_subject} | courses |",
            "|---|---|---|---|---|",
        ]
        for i, credits in enumerate(score.semester_credits):
            major = (score.major_courses_per_semester[i]
                     if i < len(score.major_courses_per_semester) else 0)
            term, year = terms[i] if i < len(terms) else ("?", "?")
            codes = ", ".join(semesters[i]) if i < len(semesters) else ""
            over_cr = (" **OVER**" if credits > ceiling
                       else (" *over ask*" if credits > asked else ""))
            over_mj = " **OVER**" if major > hard else (" *heavy*" if major > soft else "")
            lines.append(
                f"| {i + 1} | {term} {year} | {credits}{over_cr} | {major}{over_mj} | {codes} |")
        lines.append("")

    if score.assertions_passed:
        lines += ["### Did it do what the student asked?", ""]
        lines += [f"- {'PASS' if ok else 'FAIL'} — `{k}`"
                  for k, ok in sorted(score.assertions_passed.items())]
        lines.append("")
    return "\n".join(lines)


def _real_score_detail(score: RealScore, program_name: str) -> str:
    """The per-group breakdown behind `_score_detail`'s aggregate coverage fraction — same
    `RealScore` object, the other half of the same verdict. Every requirement group gets its
    own line, filled or not, rather than only the aggregate: a coverage NUMBER tells you a plan
    is short, not which of the program's own requirements it is short on.
    """
    lines: list[str] = [f"## Requirement groups: {program_name}", ""]
    lines.append(
        f"**{score.groups_satisfied}/{score.groups_total} requirement groups filled** "
        f"({score.coverage:.0%})"
    )
    lines.append("")

    if score.violations:
        counts = ", ".join(f"{k} x{v}" for k, v in sorted(score.violation_counts.items()))
        lines += [f"### Violations ({len(score.violations)}) — {counts}", ""]
        lines += [f"- `{v}`" for v in score.violations]
        lines.append("")
    else:
        lines += ["### Violations", "", "None — every hard rule held.", ""]

    lines += ["### Every requirement group", ""]
    for group in score.groups:
        mark = "✓" if group.satisfied else "✗"
        kind = "choose from list" if group.selective else "take all"
        lines.append(f"- {mark} **{group.name}** ({kind})")
        if group.filled:
            lines.append(f"    - filled: {', '.join(group.filled)}")
        if group.missing_label:
            lines.append(f"    - short: {group.missing_label}")
    lines.append("")

    return "\n".join(lines)


def load_context(root: Path) -> EvalContext:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    # MODEL_EVAL_REAL_DB_PROGRAM_ID / MODEL_EVAL_RESULTS_DIR — set once by `run.py` before
    # dispatch, which is what lets the sweep point one process at a different program and
    # results directory per school without editing config.yaml on disk. See `run.py`'s
    # `resolve_major`/`_run_sweep`.
    #
    # THERE IS NO PLAN FIXTURE ANY MORE (2026-08-12): the program id IS the selection, and the
    # fixture object every mode still takes is built from that program's own database further
    # down. See `fixtures.py`'s module docstring.
    results_dir_override = os.environ.get("MODEL_EVAL_RESULTS_DIR")
    if results_dir_override:
        cfg["paths"]["results_dir"] = results_dir_override
    # Do this BEFORE any planning: the scenarios are planned with the deterministic planner,
    # which needs to already know whether summer is schedulable.
    set_planning_terms(cfg["run"]["planning_terms"])
    questions_path = root / cfg["paths"]["questions"]
    questions = yaml.safe_load(questions_path.read_text()) if questions_path.exists() else []

    # THE MODEL'S CONTEXT, and now also the source of the students. Loaded before the prompts
    # are built, because Mode B's static block is the rendered database — its bytes are in the
    # static hash, so a stale read is a silently different experiment.
    #
    # ONE FETCH, MANY SCENARIOS. `fetch_real_db_base` is the (possibly slow) Postgres half; each
    # scenario's own database is built from it lazily, per-scenario, via `EvalContext.database_for`
    # — see `harness/real_db.py`'s module docstring for why the split exists (a real program's
    # own gen-ed menus can be far too large to render whole, so what a scenario is actually shown
    # depends on ITS `gen_ed_preference`/`world_language`, not just the program).
    #
    # THE SLOT'S context, not the server's total. `build_scenario_database` FILLS this budget —
    # a bigger number renders a bigger elective menu into Mode B's prompt — so deriving it from
    # `num_ctx` meant raising num_ctx to 65536 for four slots inflated the prompt to ~49k tokens
    # while each slot still only held 16384. Every plan request would have been truncated by the
    # context, on every model at once, and scored as if the model had rambled. `run.py check`
    # caught it (see `check_slot_context`); this is the fix. What one student is shown must be a
    # function of what one student's slot can hold.
    budget_tokens = (slot_context(cfg["run"]) - cfg["run"]["max_plan_tokens"]
                     - _PROMPT_OVERHEAD_TOKENS)
    real_db_cfg = cfg.get("real_db", {})
    pg_url = os.environ.get("MODEL_EVAL_REAL_DB_URL", real_db_cfg["url"])
    program_id = os.environ.get("MODEL_EVAL_REAL_DB_PROGRAM_ID", real_db_cfg["program_id"])
    real_db_base = fetch_real_db_base(
        pg_url, program_id,
        broaden_subjects=(tuple(real_db_cfg["broaden_subjects"])
                         if real_db_cfg.get("broaden_subjects") else None),
        force_selective_groups=tuple(real_db_cfg.get("force_selective_groups") or ()),
        manual_course_aliases=real_db_cfg.get("manual_course_aliases") or {},
    )
    print(f"[real-db] loaded program {program_id} from {pg_url} "
          f"({len(real_db_base.always_groups)} required + {len(real_db_base.elective_groups)} "
          f"elective requirement group(s), {len(real_db_base.courses_by_code)} courses in the "
          f"maximal universe)")

    # THE FIXTURE, built from the database that was just fetched — courses, requirement groups
    # and the five synthesized students. Built off the DEFAULT (no scenario preference)
    # database, the same one Mode A/QA/Explain's shared `PromptBuilder` uses: Mode A plans
    # against `fixture.catalog`, so it must be a single stable course universe rather than one
    # scenario's narrowed view.
    default_database = build_scenario_database(real_db_base, budget_tokens=budget_tokens)
    fixture = fixture_from_database(default_database)
    print(f"[fixture] {fixture.name} [{fixture.slug}] — {len(fixture.catalog)} courses, "
          f"{len(fixture.requirement_groups)} requirement groups, "
          f"{len(fixture.scenarios)} synthesized students")

    # EXTRA PROGRAMS (second majors, minors) — one fetch per distinct poid across every
    # scenario, before any scenario database is built. A failure here is raised, not skipped: a
    # second major that silently vanished would let the plan score as if the student owed
    # nothing extra, which reads as a good plan rather than a broken fixture.
    extra_program_bases: dict[str, RealDatabaseBase] = {}
    wanted = {poid for s in fixture.scenarios for poid, _kind, _label in s.additional_programs}
    if wanted:
        for poid in sorted(wanted):
            extra_id = resolve_poid(pg_url, poid)
            extra_program_bases[poid] = fetch_real_db_base(
                pg_url, extra_id,
                force_selective_groups=tuple(real_db_cfg.get("force_selective_groups") or ()),
                manual_course_aliases=real_db_cfg.get("manual_course_aliases") or {},
            )
            print(f"[real-db] + additional program poid {poid} "
                  f"({len(extra_program_bases[poid].always_groups)} required + "
                  f"{len(extra_program_bases[poid].elective_groups)} elective group(s))")

    return EvalContext(
        extra_program_bases=extra_program_bases,
        cfg=cfg,
        root=root,
        fixture=fixture,
        prompts=PromptBuilder(fixture, default_database),
        questions=questions or [],
        host=resolve_host(cfg["llamacpp"].get("host", "auto"),
                          cfg["llamacpp"].get("server_exe")),
        real_db_base=real_db_base,
        budget_tokens=budget_tokens,
    )


def sampling_options(
    cfg: dict[str, Any], *, max_tokens: int | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sampling knobs, identical for every model — plus one per-model allowance.

    `reasoning_overhead_tokens` (config.yaml, per model) is ADDED to whatever budget the caller
    asked for. It exists for models whose chat template ALWAYS opens an analysis channel that
    the server reports separately as `reasoning_content`, so those tokens are spent before a
    single character of the answer exists.

    MEASURED ON muse-glimmer-30b, 2026-08-12. Its template is Harmony-style
    (`<|start|>system<|message|>`, "Reasoning strength: high"), and `--reasoning off
    --reasoning-budget 0` does not suppress it — the server log shows `forced=0 toks` and the
    analysis runs anyway. Every plan/proposal/explain generation therefore returned
    `content: ""` with `finish_reason: length`: 43 records, zero output, scored as a total
    failure of a model that had not actually been asked a question it could answer yet. At
    3000 tokens the same request completes in 507 and returns valid JSON.

    This is a HARNESS allowance, not a handicap given to one model: every model still sees the
    same prompt, the same sampler and the same schema, and the extra budget only buys room for
    a channel the model cannot switch off. It is per-model because the overhead is a property
    of the template, not of the task.
    """
    run = cfg["run"]
    overhead = int((model_cfg or {}).get("reasoning_overhead_tokens") or 0)
    return {
        "temperature": run["temperature"],
        "top_p": run.get("top_p", 0.9),
        "seed": run["seed"],
        "max_tokens": (max_tokens or run["max_output_tokens"]) + overhead,
    }


# --- per-model lifecycle --------------------------------------------------------------------


def _warmup(ctx: EvalContext, client: LlamaCppClient, model_cfg: dict) -> None:
    for _ in range(ctx.cfg["run"]["warmup_discard"]):
        client.chat(
            WARMUP_SYSTEM, WARMUP_USER,
            options=sampling_options(ctx.cfg, max_tokens=8),
            think=model_cfg.get("think"),
        )


def _env_record(
    ctx: EvalContext, handle: ServerHandle, client: LlamaCppClient,
    model_cfg: dict, vram_baseline: list[int] | None,
) -> dict[str, Any]:
    props = client.props()
    after = gpu_memory_used_mb()
    delta = (
        [a - b for a, b in zip(after, vram_baseline)] if vram_baseline and after else None
    )
    # The server's own answer to "what context am I running?" — the check Ollama made
    # impossible. A mismatch here means this model was NOT compared at the configured size.
    server_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
    # SPECULATIVE DECODING, recorded per model because it is per model: only the entries with a
    # `draft:` speculate (see `server.resolve_draft`), so one sweep's records legitimately mix
    # drafted and undrafted models and nothing else in the file would say which is which.
    # WHETHER IT PAID OFF is answered per generation instead, by `_base_record`'s
    # `draft_accepted`/`draft_n` — an acceptance rate is a property of what was generated, not
    # of the launch, and averaging it over a model's real workload is the only honest read.
    draft_path = resolve_draft(ctx.cfg, model_cfg)
    speculative = {
        "enabled": draft_path is not None,
        "draft_gguf": str(draft_path) if draft_path else None,
        **({k: v for k, v in (ctx.cfg.get("speculative") or {}).items()
            if k in ("n_max", "n_min", "p_min")} if draft_path else {}),
    }
    return {
        "model": model_cfg["name"],
        "stage": "env",
        "gguf": model_cfg["gguf"],
        "bracket": model_cfg.get("bracket"),
        "think_setting": model_cfg.get("think"),
        "speculative": speculative,
        "argv": handle.argv,
        "vram_baseline_mb": vram_baseline,
        "vram_after_warmup_mb": after,
        "vram_delta_mb": delta,
        "offload": handle.offload(),
        "server_n_ctx": server_ctx,
        "configured_num_ctx": ctx.cfg["run"]["num_ctx"],
        # Compared against the SLOT context, not num_ctx: llama.cpp reports what one request
        # gets, which is num_ctx/parallel. Comparing to num_ctx would flag every model as
        # incomparable the moment `parallel` went above 1.
        "configured_parallel": ctx.cfg["run"].get("parallel", 1),
        # HOW MANY REQUESTS WERE IN FLIGHT alongside every generation in this run. The single
        # field that says whether this file's `ttft_s`/`total_s` are one student's latency or
        # four students queueing — above 1 they include time spent waiting for a slot, which is
        # the point, but it makes them incomparable to any `parallel: 1` run.
        "simulated_users": max(1, int(ctx.cfg["run"].get("parallel", 1))),
        "expected_slot_ctx": slot_context(ctx.cfg["run"]),
        "ctx_matches_config": (
            server_ctx == slot_context(ctx.cfg["run"]) if server_ctx else None),
        "loaded_model_id": client.loaded_model(),
        "build_info": props.get("build_info"),
    }


def _base_record(model: str, res: GenerationResult, static_hash: str, **extra) -> dict[str, Any]:
    # llama-server reports the draft counters in its own `timings` block, and ONLY when it
    # actually drafted (`draft_n > 0`) — so these keys are absent, not zero, for a model with no
    # draft, and the difference is worth preserving: "did not speculate" and "speculated and
    # every token was rejected" are opposite findings that a 0 would merge.
    timings = res.raw_final.get("timings") or {}
    drafted = timings.get("draft_n")
    rec = {
        "model": model,
        "static_hash": static_hash,
        "ttft_s": res.ttft_s,
        "total_s": res.total_s,
        "eval_count": res.eval_count,
        "prompt_eval_count": res.prompt_eval_count,
        "prompt_ms": res.prompt_ms,
        "predicted_ms": res.predicted_ms,
        "finish_reason": res.finish_reason,
        "truncated": res.truncated,
        "output": res.text,
    }
    if drafted:
        accepted = timings.get("draft_n_accepted") or 0
        rec["draft_n"] = drafted
        rec["draft_n_accepted"] = accepted
        rec["draft_acceptance"] = round(accepted / drafted, 4)
    rec.update(extra)
    return rec


# --- MODE A: revise-plan proposal (the app's production path) -------------------------------


def run_mode_a(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, mitigate: bool,
) -> None:
    """Port of ``advisor_agent.revise_plan``: propose -> apply -> re-plan -> re-validate.

    The plan that comes out is legal by construction, so viability is not the discriminator
    here — proposal groundedness and whether the student's ask was honoured are.
    """
    profile, catalog = scenario.profile, ctx.fixture.catalog
    baseline = generate_plan(profile, catalog)
    baseline_severity = severity(baseline)
    max_iterations = ctx.cfg["run"]["revise_max_iterations"] if mitigate else 1

    best_plan, best_proposal, accepted, prior_warnings = baseline, None, 0, []
    parse_failures = 0
    total_s = 0.0
    static_hash = ""
    # TTFT of the FIRST call only. Under --mitigate this stage can make several round trips,
    # but the student's wait begins once, and averaging a retry's TTFT into it would report a
    # latency nobody experiences.
    first_ttft: float | None = None
    # GENERATION TELEMETRY. This stage used to build its record by hand and drop all of it,
    # which made a whole failure mode invisible: qwen3.6-35b-a3b spent ~390 s per record
    # looping inside `rationale` until it hit the token ceiling, and `runs_baseline.jsonl`
    # showed only a large total_s next to proposal_parse_failed — no finish_reason, no
    # truncated flag, no token count, and not even the text the model produced. The raw output
    # existed solely in the transcript, so the JSONL could not answer "did it ramble or did it
    # refuse?" without opening a markdown file per record.
    #
    # MIXED SEMANTICS, on purpose, and it is why these are set here rather than via
    # _base_record: counts SUM across iterations (matching total_s, which already does),
    # `truncated` is ANY iteration hitting the ceiling (that is what explains a parse
    # failure), and finish_reason/output describe the LAST exchange (the one the transcript
    # and `proposal` correspond to).
    eval_count = prompt_eval_count = 0
    truncated = False
    finish_reason: str | None = None
    cap_source = "none"
    applied_cap: int | None = None

    last_exchange: tuple[str, str, str] = ("", "", "")
    for iteration in range(1, max_iterations + 1):
        system, user, static_hash = ctx.prompts.plan_proposal(scenario, best_plan, prior_warnings)
        try:
            res = client.chat(
                system, user,
                options=sampling_options(ctx.cfg, max_tokens=ctx.cfg["run"]["max_output_tokens"],
                                         model_cfg=model_cfg),
                think=model_cfg.get("think"),
                response_format=json_response_format(
                    "PlanEditProposal",
                    proposal_schema(scenario.profile.remaining_courses,
                                    catalog_tags(ctx.fixture.catalog))),
            )
        except LlamaCppError as exc:
            out.write(json.dumps({
                "model": model_cfg["name"], "stage": "plan_mode_a", "scenario": scenario.id,
                "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
            return
        total_s += res.total_s
        if first_ttft is None:
            first_ttft = res.ttft_s
        eval_count += res.eval_count or 0
        prompt_eval_count += res.prompt_eval_count or 0
        truncated = truncated or bool(res.truncated)
        finish_reason = res.finish_reason
        last_exchange = (system, user, res.text)

        parsed = scorers.extract_json(res.text)
        if parsed is None:
            parse_failures += 1
            # The app treats an unusable proposal as "keep the best legal plan so far" — so
            # does the harness. A model that always fails here still returns a working plan to
            # the student; it just never helps. That distinction is the point of Mode A.
            break

        proposal = Proposal(
            rationale=str(parsed.get("rationale") or ""),
            reorder=[str(c) for c in parsed.get("reorder") or []],
            defer=[str(c) for c in parsed.get("defer") or []],
            avoid_tags=[str(t) for t in parsed.get("avoid_tags") or []],
            max_credits_per_semester=parsed.get("max_credits_per_semester"),
        )
        # Port of the app's deterministic backstop. Recorded separately from the model's own
        # answer on purpose: `proposal_sets_credit_cap` must keep measuring whether the MODEL
        # understood the request, while the student still gets the load they asked for. Reading
        # those two columns together is what says "the fallback is carrying this model".
        if proposal.max_credits_per_semester is None:
            asked = extract_credit_cap(scenario.feedback)
            if asked is not None:
                proposal.max_credits_per_semester = asked
                cap_source = "text"
        else:
            cap_source = "model"
        revised = generate_plan(apply_proposal(profile, proposal, catalog), catalog)
        best_plan, best_proposal, accepted = revised, parsed, iteration
        applied_cap = proposal.max_credits_per_semester
        if severity(revised) <= baseline_severity:
            break
        prior_warnings = list(revised.warnings) + [
            w for s in revised.semesters for w in s.warnings
        ]

    semesters = plan_scorers.plan_to_semesters(best_plan)
    # The APPLIED cap is what the student actually experiences, so score against it, not
    # against the profile's original and not against what the model said. Those differ whenever
    # the deterministic backstop filled a cap the model left null: the plan really was built at
    # the tighter number, so scoring it against the looser one would grade a plan nobody got.
    effective = profile
    if applied_cap:
        effective = profile.replace(max_credits_per_semester=int(applied_cap))
    score = plan_scorers.score_plan(ctx.fixture, effective, semesters)
    score.assertions_passed = plan_scorers.check_assertions(
        scenario.assertions, semesters, score, proposal=best_proposal or {},
        canonical=ctx.fixture.canonical,
    )

    planned_codes = {plan_scorers.normalize_code(c) for s in semesters for c in s}
    rec = {
        "model": model_cfg["name"],
        "stage": "plan_mode_a",
        "scenario": scenario.id,
        "run_idx": run_idx,
        "mitigated": mitigate,
        "expect_unsatisfiable": scenario.expect_unsatisfiable,
        "static_hash": static_hash,
        "fixture_hash": ctx.fixture.fixture_hash,
        "ttft_s": first_ttft,
        "total_s": total_s,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "output": last_exchange[2],
        "iterations": accepted,
        "credit_cap_source": cap_source,
        "credit_cap_from_text": extract_credit_cap(scenario.feedback),
        "proposal_parse_failed": best_proposal is None,
        "parse_failures": parse_failures,
        "proposal": best_proposal,
        "baseline_unplanned": baseline_severity,
        "revised_unplanned": severity(best_plan),
        "plan_not_worse": severity(best_plan) <= baseline_severity,
        "semesters": semesters,
        **score.as_record(),
    }
    if best_proposal is not None:
        rec.update(plan_scorers.score_proposal(best_proposal, profile, ctx.fixture))
        rec["rationale_flags"] = plan_scorers.rationale_flags(
            str(best_proposal.get("rationale") or ""), planned_codes, ctx.fixture.by_code
        )
    out.write(json.dumps(rec, default=str) + "\n")
    ctx.write_transcript(
        model_cfg["name"], "plan_mode_a", scenario.id, run_idx, *last_exchange,
        verdict=f"grounded={rec.get('proposal_grounded')} "
                f"viable={rec.get('plan_viable')} cap_source={cap_source}",
        detail=_score_detail(score, semesters, effective),
    )


# --- MODES B and C: free-form plan of study --------------------------------------------------
#
# ONE function, two prompts. Mode C exists to measure what the published sample plan buys, and
# that subtraction is only valid if everything downstream of the prompt — sampling options, the
# response grammar, the parse, the scorer, the record shape — is identical. Two near-copies of
# this function would drift, and the drift would land in the delta.


def run_mode_b(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, mitigate: bool,
) -> None:
    """The model emits the whole schedule. Scored against the real database it was shown
    (`real_scoring.score_against_real_db`) — not the fixture; see that module's docstring.

    This is where models actually separate. A plan is viable only if it has ZERO hard
    violations and covers every CHECKABLE requirement — one prerequisite mistake anywhere in
    eight semesters fails the whole plan, which is the correct standard: a student following it
    would be turned away at registration.
    """
    _run_freeform_plan(
        ctx, client, model_cfg, scenario, run_idx, out,
        mitigate=mitigate, stage="plan_mode_b",
        prompt=ctx.prompts_for(scenario).plan_freeform(scenario),
    )


def run_mode_b_thinking(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, thinking_budget: int,
) -> None:
    """Mode B, but the model reasons BEFORE it commits to the plan, bounded to
    `thinking_budget` tokens — see `run_thinking_experiment` for why this needs its own server
    process and can never run inside the baseline's own `run_mode_b` pass.

    This is a different mechanism from the removed post-hoc `rationale` field
    (`prompts.PLAN_SCHEMA`'s own history note): that text was written AFTER `semesters` was
    already fixed, so nothing in it could have changed the plan — pure narration, correctly
    cut for the tokens it cost. Reasoning that happens before the grammar-constrained JSON
    starts is structurally different: it is generated BEFORE any course is chosen, so it is at
    least possible for it to change what gets picked. Whether it actually does, on THESE
    models, is the open question this experiment measures — same prompt, same schema, same
    scorer as `run_mode_b`, the only difference is reasoning on vs. off.
    """
    _run_freeform_plan(
        ctx, client, model_cfg, scenario, run_idx, out,
        mitigate=False, stage="plan_mode_b_thinking",
        prompt=ctx.prompts_for(scenario).plan_freeform(scenario),
        think_override=True, extra_max_tokens=thinking_budget,
    )


def _run_freeform_plan(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, mitigate: bool,
    stage: str, prompt: tuple[str, str, str],
    think_override: bool | None = None, extra_max_tokens: int = 0,
) -> None:
    """`think_override`/`extra_max_tokens` exist for exactly one caller, `run_mode_b_thinking`:
    force reasoning ON for this call (independent of `model_cfg["think"]`, which stays the
    baseline's own setting) and widen the token budget by the reasoning budget the server was
    launched with, so a bounded pre-plan reasoning pass doesn't starve the JSON plan that has
    to follow it out of `max_plan_tokens`. Both are no-ops (`None`/`0`) on every other call.
    """
    system, user, static_hash = prompt
    database = ctx.database_for(scenario)
    try:
        res = client.chat(
            system, user,
            options=sampling_options(
                ctx.cfg, max_tokens=ctx.cfg["run"]["max_plan_tokens"] + extra_max_tokens,
                model_cfg=model_cfg),
            think=think_override if think_override is not None else model_cfg.get("think"),
            # THE COURSE ENUM MUST COME FROM THIS SCENARIO'S OWN `database`, NOT
            # `ctx.fixture.catalog` AND NOT ANOTHER SCENARIO'S DATABASE. This schema is what
            # grammar-constrains the model's OUTPUT — get it wrong and the model is not merely
            # scored against the wrong catalog, it is physically unable to emit any course
            # outside whatever catalog built the enum, no matter what program or elective menu
            # the PROMPT TEXT showed it. That was exactly backwards for a non-CS `--major`: the
            # model was shown a real WGSS catalog and grammar-forced to answer only in CS
            # Machine Intelligence course codes, which is why every course it could possibly
            # write scored `hallucinated_course` against the real program — the grammar had
            # already made a correct answer impossible before the model wrote a token. The same
            # failure mode now applies PER SCENARIO, not just per program.
            response_format=json_response_format(
                "PlanOfStudy", plan_schema(
                    ctx.cfg["run"]["planning_terms"],
                    sorted(database.course_codes))),
        )
    except LlamaCppError as exc:
        out.write(json.dumps({
            "model": model_cfg["name"], "stage": stage, "scenario": scenario.id,
            "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
        return

    parsed = scorers.extract_json(res.text)
    rec = _base_record(
        model_cfg["name"], res, static_hash,
        stage=stage, scenario=scenario.id, run_idx=run_idx, mitigated=mitigate,
        expect_unsatisfiable=scenario.expect_unsatisfiable,
        fixture_hash=ctx.fixture.fixture_hash,
    )

    if not parsed or not isinstance(parsed.get("semesters"), list):
        # Not a scoring edge case — a plan that cannot be parsed cannot be shown to a student.
        rec.update(RealScore().as_record())
        rec["structure_ok"] = False
        rec["parse_failure_reason"] = "no JSON object" if not parsed else "no semesters array"
        out.write(json.dumps(rec, default=str) + "\n")
        ctx.write_transcript(model_cfg["name"], stage, scenario.id, run_idx,
                             system, user, res.text,
                             verdict=f"UNPARSEABLE ({rec['parse_failure_reason']})")
        return

    raw_semesters = [
        [str(c) for c in (s.get("courses") or [])]
        for s in parsed["semesters"] if isinstance(s, dict)
    ]
    # Every real course maps to itself except the aliased subset, which maps to its primary —
    # same shape as `fixture.canonical`, built from `database` instead.
    alias_of = {row["course_code"]: row["alias_of"] for row in database.rows("course_aliases")}
    real_canonical = {code: alias_of.get(code, code) for code in database.course_codes}
    canon = lambda code: real_canonical.get(code, code)  # noqa: E731
    completed = {canon(plan_scorers.normalize_code(c)) for c in scenario.profile.completed_courses}
    # DEDUPE BEFORE SCORING, not after — see plan_scorers.dedupe_semesters's docstring. A
    # repeated or already-completed course is deleted here so `score_against_real_db` never
    # sees it and `plan_viable` is never held hostage by one trivially-fixable repeated line.
    semesters, duplicates_removed = plan_scorers.dedupe_semesters(
        raw_semesters, canon=canon, completed=completed
    )
    # SCORED AGAINST `database` — this SCENARIO's own real, live catalog, the one it was
    # actually shown and grammar-constrained to — not the hand-authored fixture, and not any
    # other scenario's database. See real_scoring.py's module docstring for the incidents (real
    # courses flagged hallucinated, invented gen-ed groups counted as "missing") that made
    # scoring Mode B against the fixture wrong rather than merely approximate; the same
    # reasoning is why this must be `database`, not `ctx.database_for` of a different scenario.
    real_score = score_against_real_db(
        database, scenario.profile, semesters, assertions=scenario.assertions
    )
    planned_codes = {plan_scorers.normalize_code(c) for s in semesters for c in s}
    rec.update(real_score.as_record())
    rec.update({
        "semesters": semesters,
        "semester_count": len(semesters),
        "over_horizon": len(semesters) > scenario.profile.semesters_to_plan,
        "duplicates_removed": duplicates_removed,
        "declared_unplanned": parsed.get("unplanned") or [],
        "rationale": parsed.get("rationale"),
        "rationale_flags": plan_scorers.rationale_flags(
            str(parsed.get("rationale") or ""), planned_codes, database.course_codes
        ),
        "term_labels_ok": _term_labels_ok(parsed["semesters"], scenario.profile),
        # Empty on every call except `run_mode_b_thinking`'s (see that function and
        # `_run_freeform_plan`'s own docstring) — reasoning stays off at launch everywhere
        # else, so this channel never fires there.
        "reasoning_chars": len(res.reasoning_text),
    })
    out.write(json.dumps(rec, default=str) + "\n")
    program_rows = database.rows("programs")
    program_name = program_rows[0]["name"] if program_rows else "(unknown program)"
    transcript_text = res.text
    if res.reasoning_text:
        transcript_text = f"<think>\n{res.reasoning_text}\n</think>\n\n{res.text}"
    ctx.write_transcript(
        model_cfg["name"], stage, scenario.id, run_idx, system, user, transcript_text,
        verdict=f"viable={real_score.viable} coverage={real_score.requirement_coverage:.0%} "
                f"violations={real_score.violation_counts or 'none'} | "
                f"{real_score.groups_satisfied}/{real_score.groups_total} requirement groups",
        detail=_score_detail(real_score, semesters, scenario.profile)
              + "\n" + _real_score_detail(real_score, program_name),
    )


def _term_labels_ok(semesters: list[dict], profile: Profile) -> bool:
    """Did the model label its own semesters with the right term/year sequence?

    Diagnostic only — scoring uses positional order, so a model that gets the calendar wrong
    but the sequencing right is not punished twice for one mistake.
    """
    from .planner import next_term
    term, year = profile.start_term.lower(), profile.start_year
    for entry in semesters:
        if not isinstance(entry, dict):
            return False
        if str(entry.get("term", "")).lower() != term or entry.get("year") != year:
            return False
        term, year = next_term(term, year)
    return True


# --- grounded QA + explain-plan --------------------------------------------------------------


def run_qa(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    question: dict, run_idx: int, out, *, mitigate: bool,
) -> None:
    chunks = question.get("context") or []
    system, user, static_hash = ctx.prompts.grounded_qa(question["question"], chunks)
    try:
        res = client.chat(
            system, user,
            options=sampling_options(ctx.cfg, model_cfg=model_cfg),
            think=model_cfg.get("think"),
        )
    except LlamaCppError as exc:
        out.write(json.dumps({
            "model": model_cfg["name"], "stage": "qa", "question_id": question["id"],
            "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
        return

    rec = _base_record(
        model_cfg["name"], res, static_hash,
        stage="qa", question_id=question["id"], category=question.get("category"),
        # `topic` (course | policy | process) is what `report._qa_section`'s by-topic table
        # slices on — a bot can be fluent about prerequisites and useless about how to CODO,
        # and those two failures average into a single QA number that describes neither.
        topic=question.get("topic"),
        expected_behavior=question["expected_behavior"], run_idx=run_idx, mitigated=mitigate,
        context_json=json.dumps(chunks, sort_keys=True),
    )
    rec.update(scorers.score_qa(res.text, chunks, question["expected_behavior"]))
    out.write(json.dumps(rec, default=str) + "\n")
    ctx.write_transcript(
        model_cfg["name"], "qa", question["id"], run_idx, system, user, res.text,
        verdict=f"expected={question['expected_behavior']} ok={rec.get('behavior_ok')} "
                f"faith_flags={len(rec.get('faithfulness_flags') or [])}",
    )


def run_explain(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, mitigate: bool,
) -> None:
    """Explain the deterministic baseline plan. Every model explains the SAME plan — feeding
    each model its own Mode B plan would confound explanation quality with planning quality."""
    plan = generate_plan(scenario.profile, ctx.fixture.catalog)
    # Shape this exactly like ``PlanResponse.model_dump_json()`` — the router hands the model
    # ``PlannedCourse`` objects (code + title + credits), NOT bare course codes. Sending codes
    # alone was a harness-only bug with a measurable cost: with no title in the payload, models
    # fill the gap from parametric memory and tell the student CS 35400 is "Theory of
    # Computation" (it is Operating Systems). That failure mode is an artifact of the harness,
    # not something production can produce, so scoring it was scoring a fiction.
    by_code = {c.code: c for c in ctx.fixture.catalog}
    def _planned_course(code: str) -> dict:
        course = by_code.get(code)
        if course is None:  # a plan can only hold catalog codes; be explicit if that breaks
            return {"code": code, "title": "", "credits": 0}
        return {"code": course.code, "title": course.title, "credits": course.credits}

    plan_json = json.dumps(
        {
            "student_name": scenario.profile.name,
            "degree_program": scenario.profile.degree_program,
            "semesters": [
                {"term": s.term, "year": s.year,
                 "courses": [_planned_course(code) for code in s.courses],
                 "total_credits": s.total_credits, "warnings": s.warnings}
                for s in plan.semesters
            ],
            "unplanned_courses": plan.unplanned_courses,
            "warnings": plan.warnings,
        },
        indent=2, sort_keys=True,
    )
    question = ctx.cfg["run"].get("explain_question", "Why is my plan sequenced this way?")
    system, user, static_hash = ctx.prompts.explain_plan(question, plan_json)
    try:
        res = client.chat(
            system, user,
            options=sampling_options(
                ctx.cfg, max_tokens=ctx.cfg["run"].get("max_explain_tokens"),
                model_cfg=model_cfg),
            think=model_cfg.get("think")
        )
    except LlamaCppError as exc:
        out.write(json.dumps({
            "model": model_cfg["name"], "stage": "explain", "scenario": scenario.id,
            "run_idx": run_idx, "mitigated": mitigate, "error": str(exc)}) + "\n")
        return

    rec = _base_record(
        model_cfg["name"], res, static_hash,
        stage="explain", scenario=scenario.id, run_idx=run_idx, mitigated=mitigate,
        fixture_hash=ctx.fixture.fixture_hash,
    )
    rec["faithfulness_flags"] = scorers.faithfulness_flags(res.text, plan_json)
    # CLAIMS THE PLAN ITSELF CONTRADICTS — see `scorers.explain_flags`. Faithfulness alone only
    # asks "did it invent a course code"; an explanation can pass that and still tell a student
    # a course sits in a semester it does not, or describe a plan as complete while the planner
    # is reporting courses it could not fit. Both are checkable against the payload the model
    # was handed, because the harness generated it.
    rec["explain_flags"] = scorers.explain_flags(
        res.text, plan_scorers.plan_to_semesters(plan), list(plan.unplanned_courses)
    )
    rec["explain_auto_pass"] = not rec["explain_flags"] and not rec["faithfulness_flags"]
    rec["unplanned_count"] = len(plan.unplanned_courses)
    rec["needs_review"] = True
    out.write(json.dumps(rec, default=str) + "\n")
    ctx.write_transcript(
        model_cfg["name"], "explain", scenario.id, run_idx, system, user, res.text,
        verdict=f"faith_flags={len(rec['faithfulness_flags'])} "
                f"explain_flags={len(rec['explain_flags'])} truncated={res.truncated}",
        detail=("## Explanation checks\n\n" + (
            "\n".join(f"- ⚠️ {f}" for f in rec["explain_flags"] + rec["faithfulness_flags"])
            or "None — every checkable claim matches the plan."
        )),
    )


# --- top level --------------------------------------------------------------------------------


class _LockedWriter:
    """`out`, made safe for the concurrent workers below.

    Every task function (`run_mode_a`, `run_mode_b`, `run_explain`, `run_qa`) writes its record
    with a bare `out.write(json.dumps(rec) + "\\n")`. With four of them running at once those
    calls interleave mid-line and corrupt the JSONL — the identical failure `_run_lock` exists
    to prevent between processes, arriving from inside one process instead. Wrapping the file
    rather than adding a lock to each call site keeps all four functions unchanged and means a
    task function added later cannot forget to take it.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._stream.write(text)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()


def _item_name(item: Any) -> str:
    """A short, readable id for an error line — scenarios carry `.id`, questions are dicts."""
    if isinstance(item, dict):
        return str(item.get("id") or item.get("question") or "?")[:60]
    return str(getattr(item, "id", item))[:60]


def _run_pool(items: list, work, *, concurrency: int, label: str) -> None:
    """Run `work(item)` over `items` with exactly `concurrency` in flight — the simulation.

    A WORKER POOL, NOT A BATCH DUMP. Handing all 8 scenarios to the server at once and letting
    it drain them 4 at a time would measure a backlog draining, which is a different question:
    the wait a student experiences depends on how many people are on the site, not on how deep
    the queue behind them is. A pool of `concurrency` keeps the in-flight count at exactly the
    number of simulated users for the whole run.

    ONE ITEM'S FAILURE IS ONE ITEM'S FAILURE. Serially, an exception propagated out of
    `run_model` and abandoned the model; that is no longer acceptable when three other requests
    are mid-flight and would be cancelled with it, so each item's exception is caught, recorded
    as a `stage: "error"` row, and the rest of the wave continues.
    """
    if concurrency <= 1:
        for item in items:
            work(item)
        return
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=label) as pool:
        futures = {pool.submit(work, item): item for item in items}
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                print(f"[error] {label} {_item_name(futures[future])}: "
                      f"{type(exc).__name__}: {exc}")


def _ensure_server_alive(
    ctx: EvalContext, handle: ServerHandle, client: LlamaCppClient, model_cfg: dict,
    vram_baseline: list[int] | None,
) -> tuple[ServerHandle, LlamaCppClient, dict[str, Any] | None]:
    """Cheap health check before every scenario; if the server died, restart it and warm it
    back up so the REST of this model's scenarios still run instead of every one after the
    crash failing identically. Real incident 2026-08-06: llama-server hit the same SIGABRT
    (confirmed via `dmesg`, not a harness bug — the crash address was identical across three
    occurrences over two full restarts) partway through a model's run, at a different scenario
    each time, always with the GPU near its VRAM ceiling. Nothing before this checked the
    server was still up, so every scenario after the crash failed with "connection refused"
    and the rest of that model's data was simply lost. This does not fix the crash — it stops
    one crash from costing more than the one request it happened on.

    Returns `(handle, client, recovery_env_record)` — the THIRD element is `None` on the
    common path (server was fine) and an `_env_record`-shaped dict (`stage:
    "env_recovered"`) when a restart happened, for the caller to log. Raises `RuntimeError` if
    the restart attempt itself fails (GPU wedged, model won't reload) — that is a real reason
    to stop trying this model, not to retry silently.
    """
    try:
        client.props()
        return handle, client, None
    except LlamaCppError:
        pass
    print(f"[recover] {model_cfg['name']}: server unresponsive — restarting")
    try:
        handle.stop(trim_log_lines=ctx.cfg["llamacpp"].get("keep_log_lines", 400))
    except Exception:  # noqa: BLE001 — the old process is already gone; nothing to salvage
        pass
    time.sleep(ctx.cfg["llamacpp"].get("cooldown_s", 5))
    try:
        new_handle = start_server(ctx.cfg, model_cfg, ctx.log_dir, host=ctx.host)
        new_client = LlamaCppClient(new_handle.base_url,
                                    timeout_s=ctx.cfg["run"]["request_timeout_s"])
        _warmup(ctx, new_client, model_cfg)
    except (RuntimeError, TimeoutError) as exc:
        raise RuntimeError(f"recovery restart failed: {exc}") from exc
    env = _env_record(ctx, new_handle, new_client, model_cfg, vram_baseline)
    env["stage"] = "env_recovered"
    return new_handle, new_client, env


def run_model(ctx: EvalContext, model_cfg: dict, out, *, tasks: set[str], mitigate: bool) -> None:
    name = model_cfg["name"]
    vram_baseline = gpu_memory_used_mb()
    try:
        handle = start_server(ctx.cfg, model_cfg, ctx.log_dir, host=ctx.host)
    except (RuntimeError, TimeoutError) as exc:
        print(f"[SKIP] {name}: {exc}")
        out.write(json.dumps({"model": name, "stage": "error", "error": str(exc)}) + "\n")
        out.flush()
        return

    client = LlamaCppClient(handle.base_url, timeout_s=ctx.cfg["run"]["request_timeout_s"])
    try:
        _warmup(ctx, client, model_cfg)
        env = _env_record(ctx, handle, client, model_cfg, vram_baseline)
        out.write(json.dumps(env, default=str) + "\n")
        out.flush()
        print(f"[env] {name}: vram_delta={env['vram_delta_mb']} offload={env['offload']} "
              f"n_ctx={env['server_n_ctx']}")
        if env["ctx_matches_config"] is False:
            print(f"[WARN] {name}: server reports n_ctx={env['server_n_ctx']} per slot, config "
                  f"implies {env['expected_slot_ctx']} (num_ctx {env['configured_num_ctx']} / "
                  f"parallel {env['configured_parallel']}) — NOT comparable to the others.")

        # HOW MANY STUDENTS ARE ON THE SITE AT ONCE — the same number as the server's slot
        # count, deliberately, and read from the one place that already sets it. Diverging the
        # two would either leave paid-for slots idle (concurrency < parallel) or queue requests
        # behind a full server, which measures a backlog rather than concurrent users
        # (concurrency > parallel). See config.yaml's `run.parallel`.
        concurrency = max(1, int(ctx.cfg["run"].get("parallel", 1)))
        out = _LockedWriter(out)

        # BUILT BEFORE ANY WORKER STARTS. `database_for`/`prompts_for` memoize per scenario, and
        # a scenario's database is a token-budget fill loop, not a cheap dict lookup. Warming
        # them here keeps that work out of the measured window — otherwise the first wave's
        # latency would include building four databases and would not be comparable to the
        # second wave's, which finds them cached.
        for scenario in ctx.fixture.scenarios:
            ctx.prompts_for(scenario)

        runs = ctx.cfg["run"]["runs_per_pair"]
        for run_idx in range(runs):
            # HEALTH-CHECKED PER WAVE, NOT PER SCENARIO. Serially this ran before every
            # scenario; under a pool it cannot, because a restart swaps `handle`/`client` out
            # from under any worker already using them. Once per wave keeps the recovery
            # behaviour (a crash still costs one wave, not the model's whole run) without
            # racing the workers for the server it is trying to replace.
            try:
                handle, client, recovery = _ensure_server_alive(
                    ctx, handle, client, model_cfg, vram_baseline)
            except RuntimeError as exc:
                print(f"[SKIP] {name}: {exc} — abandoning remaining scenarios for this model")
                out.write(json.dumps({"model": name, "stage": "error", "error": str(exc)}) + "\n")
                out.flush()
                return
            if recovery:
                out.write(json.dumps(recovery, default=str) + "\n")
                out.flush()
                print(f"[recover] {name}: back up, resuming at replicate {run_idx + 1}")

            def one_scenario(scenario, *, _client=client, _run_idx=run_idx) -> None:
                """One simulated student's session: their plan, then the explanation of it.

                The tasks stay SEQUENTIAL within a scenario — a student does not ask for a plan
                and an explanation of it simultaneously. The concurrency is across students,
                which is what four people on the site actually looks like.
                """
                if "plan_a" in tasks:
                    run_mode_a(ctx, _client, model_cfg, scenario, _run_idx, out, mitigate=mitigate)
                if "plan_b" in tasks:
                    run_mode_b(ctx, _client, model_cfg, scenario, _run_idx, out, mitigate=mitigate)
                if "explain" in tasks:
                    run_explain(ctx, _client, model_cfg, scenario, _run_idx, out, mitigate=mitigate)
                out.flush()

            # Plan first: the feature that matters most gets the data if a run is cut short.
            _run_pool(list(ctx.fixture.scenarios), one_scenario,
                      concurrency=concurrency, label="scenario")

            if "qa" in tasks:
                try:
                    handle, client, recovery = _ensure_server_alive(
                        ctx, handle, client, model_cfg, vram_baseline)
                except RuntimeError as exc:
                    print(f"[SKIP] {name}: {exc} — abandoning QA for this model")
                    out.write(json.dumps({"model": name, "stage": "error", "error": str(exc)}) + "\n")
                    out.flush()
                    return
                if recovery:
                    out.write(json.dumps(recovery, default=str) + "\n")
                    out.flush()
                    print(f"[recover] {name}: back up, resuming QA")

                def one_question(question, *, _client=client, _run_idx=run_idx) -> None:
                    run_qa(ctx, _client, model_cfg, question, _run_idx, out, mitigate=mitigate)

                _run_pool(list(ctx.questions), one_question,
                          concurrency=concurrency, label="qa")
                out.flush()
            print(f"  [{name}] replicate {run_idx + 1}/{runs} done "
                  f"(concurrency {concurrency})")
    finally:
        handle.stop(trim_log_lines=ctx.cfg["llamacpp"].get("keep_log_lines", 400))
        time.sleep(ctx.cfg["llamacpp"].get("cooldown_s", 5))


class ConcurrentRunError(RuntimeError):
    """Another run already owns this results directory."""


@contextmanager
def _run_lock(results_dir: Path, tag: str):
    """Refuse to start a second run against the same results file.

    Two runs appending to one JSONL interleave mid-line and corrupt it, and — worse — they
    fight over the GPU and over llama-server's port, so both sets of latency numbers become
    garbage while still looking plausible. This was not hypothetical: it happened during
    development, and the only visible symptom was a results file containing a model nobody
    had asked for. A stale lock (previous run killed) is reported with the PID so it can be
    cleared deliberately rather than silently stomped.
    """
    lock = results_dir / f".{tag}.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        alive = pid > 0 and Path(f"/proc/{pid}").exists()
        raise ConcurrentRunError(
            f"{lock} exists (pid {pid or 'unknown'}, "
            f"{'STILL RUNNING' if alive else 'stale — process is gone'}). "
            + (
                "Stop that run before starting another; concurrent runs corrupt "
                f"runs_{tag}.jsonl and contend for the GPU."
                if alive else f"Delete {lock} to proceed."
            )
        )
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_eval(
    root: Path, *, models: list[str] | None = None, brackets: list[str] | None = None,
    tasks: list[str] | None = None, mitigate: bool = False,
) -> Path:
    ctx = load_context(root)
    tag = "mitigated" if mitigate else "baseline"
    selected = [
        m for m in ctx.cfg["models"]
        if (not models or m["name"] in models) and (not brackets or m.get("bracket") in brackets)
    ]
    if not selected:
        raise SystemExit("No models matched the filter.")

    task_set = set(tasks or ctx.cfg["run"]["default_tasks"])
    if not task_set:
        raise SystemExit("No tasks left to run.")
    out_path = ctx.results_dir / f"runs_{tag}.jsonl"

    # The lock is taken BEFORE anything is written. Taking it later was a real bug: a run that
    # lost the race had already overwritten meta_{tag}.json, so the metadata described a run
    # whose records were never produced — and the results file looked like it contained models
    # nobody had asked for.
    with _run_lock(ctx.results_dir, tag):
        write_meta(ctx, tag, selected, task_set)
        mode = "a" if out_path.exists() else "w"
        print(f"[run] {len(selected)} model(s), tasks={sorted(task_set)} -> {out_path} "
              f"(mode={mode})")
        with out_path.open(mode) as out:
            for model_cfg in selected:
                print(f"\n=== {model_cfg['name']} ({model_cfg.get('bracket')}) ===")
                run_model(ctx, model_cfg, out, tasks=task_set, mitigate=mitigate)
    return out_path


def run_model_thinking(
    ctx: EvalContext, model_cfg: dict, out, *, thinking_budget: int,
) -> None:
    """One model, Mode B only, reasoning ON — the thinking-experiment analogue of `run_model`.

    DELIBERATELY NARROWER than `run_model`: no plan_a/qa/explain (this experiment is about
    Mode B specifically — see `run_mode_b_thinking`'s own docstring for why), and no
    `_ensure_server_alive` recovery loop, since this pass is short enough (one task, one
    scenario list) that a crash mid-run is better surfaced as a failure than silently patched
    over with a fresh warmup — an experimental result should not quietly absorb a restart the
    way a multi-hour baseline sweep has to.
    """
    name = model_cfg["name"]
    vram_baseline = gpu_memory_used_mb()
    try:
        handle = start_server(ctx.cfg, model_cfg, ctx.log_dir, host=ctx.host,
                              thinking_budget=thinking_budget)
    except (RuntimeError, TimeoutError) as exc:
        print(f"[SKIP] {name}: {exc}")
        out.write(json.dumps({"model": name, "stage": "error", "error": str(exc)}) + "\n")
        out.flush()
        return

    client = LlamaCppClient(handle.base_url, timeout_s=ctx.cfg["run"]["request_timeout_s"])
    try:
        _warmup(ctx, client, model_cfg)
        env = _env_record(ctx, handle, client, model_cfg, vram_baseline)
        env["thinking_budget"] = thinking_budget
        out.write(json.dumps(env, default=str) + "\n")
        out.flush()
        print(f"[env] {name}: vram_delta={env['vram_delta_mb']} offload={env['offload']} "
              f"n_ctx={env['server_n_ctx']} thinking_budget={thinking_budget}")

        runs = ctx.cfg["run"]["runs_per_pair"]
        for run_idx in range(runs):
            for scenario in ctx.fixture.scenarios:
                run_mode_b_thinking(ctx, client, model_cfg, scenario, run_idx, out,
                                    thinking_budget=thinking_budget)
                out.flush()
            print(f"  [{name}] replicate {run_idx + 1}/{runs} done")
    finally:
        handle.stop(trim_log_lines=ctx.cfg["llamacpp"].get("keep_log_lines", 400))
        time.sleep(ctx.cfg["llamacpp"].get("cooldown_s", 5))


def run_thinking_experiment(
    root: Path, *, models: list[str] | None = None, brackets: list[str] | None = None,
) -> Path:
    """Top level for the Mode B pre-plan-reasoning experiment (`--tasks plan_b_thinking`).

    OWN SERVER, OWN RESULTS FILE (`runs_thinking.jsonl`), OWN LOCK — never `run_eval`'s
    `runs_baseline.jsonl`. The baseline launches every model with `--reasoning off
    --reasoning-budget 0`; this launches with reasoning ON and `thinking.budget_tokens` (see
    config.yaml). Those are two different `argv`s for two different server processes, so
    keeping the results apart is what makes "reasoning on vs. off, same prompt, same schema,
    same scorer" a comparison that means something, rather than one file mixing two settings.

    NOT in `run.default_tasks` — this only runs when asked for by name, exactly like
    `converge` used to before it was folded into `--tasks`, except this one stays opt-in on
    purpose: it is a genuine experiment (does reasoning change the plan?), not a mode the app
    ships, and every sweep paying its cost by default would be measuring something nobody asked
    for.
    """
    ctx = load_context(root)
    thinking_cfg = ctx.cfg.get("thinking") or {}
    if not thinking_cfg:
        raise SystemExit(
            "config.yaml has no `thinking:` block — add `thinking: {budget_tokens: N}` to use "
            "--tasks plan_b_thinking."
        )
    budget = int(thinking_cfg.get("budget_tokens", 512))

    selected = [
        m for m in ctx.cfg["models"]
        if (not models or m["name"] in models) and (not brackets or m.get("bracket") in brackets)
    ]
    if not selected:
        raise SystemExit("No models matched the filter.")

    out_path = ctx.results_dir / "runs_thinking.jsonl"
    with _run_lock(ctx.results_dir, "thinking"):
        write_meta(ctx, "thinking", selected, {"plan_b_thinking"})
        mode = "a" if out_path.exists() else "w"
        print(f"[run] {len(selected)} model(s), thinking_budget={budget} -> {out_path} "
              f"(mode={mode})")
        with out_path.open(mode) as out:
            for model_cfg in selected:
                print(f"\n=== {model_cfg['name']} (thinking, budget={budget}) ===")
                run_model_thinking(ctx, model_cfg, out, thinking_budget=budget)
    return out_path


def write_meta(ctx: EvalContext, tag: str, selected: list[dict], tasks: set[str]) -> None:
    scenario = ctx.fixture.scenarios[0]
    meta = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": ctx.host,
        "tasks": sorted(tasks),
        "models": [m["name"] for m in selected],
        # See `_env_record`'s `simulated_users`. Hoisted to the top level of the meta too,
        # because "is this a load-test run or a solo run?" is the first thing that has to be
        # true before any latency number in the matching JSONL means anything.
        "simulated_users": max(1, int(ctx.cfg["run"].get("parallel", 1))),
        "run": ctx.cfg["run"],
        "llamacpp": ctx.cfg["llamacpp"],
        # WHICH SAMPLING PATH THIS FILE'S RECORDS CAME OFF. A speculative run reproduces the
        # unspeculated one only approximately (see `server.py`'s speculative section), so this
        # is the field that says whether two results directories may be pooled — `enabled` here
        # is the effective value after `--spec`/`--no-spec`, not just what config.yaml says.
        "speculative": {
            **(ctx.cfg.get("speculative") or {}),
            "enabled": speculative_enabled(ctx.cfg),
            "models": {m["name"]: (str(d) if (d := resolve_draft(ctx.cfg, m)) else None)
                       for m in selected},
        },
        # THE PROGRAM this run evaluated. Still keyed "fixture" so every reader written
        # against older meta files keeps working; there is no fixture FILE any more (see
        # `fixtures.py`), so `path` is the database's own identity and `hash` its db_hash.
        "fixture": {
            "path": str(ctx.fixture.path.name),
            "name": ctx.fixture.name,
            "slug": ctx.fixture.slug,
            "program_id": ctx.fixture.real_db_program_id,
            "hash": ctx.fixture.fixture_hash,
            "verified": ctx.fixture.verified,
            "scenarios": [s.id for s in ctx.fixture.scenarios],
            "courses": len(ctx.fixture.catalog),
            "requirement_groups": len(ctx.fixture.requirement_groups),
        },
        # Mode B's whole context — ONE PER SCENARIO now that each scenario's own
        # gen_ed_preference/world_language narrows the real database differently (see
        # `harness/real_db.py`'s module docstring). Recorded per table, not just as one hash:
        # when an export changes, "which table moved" is the first question, and a single
        # digest cannot answer it. Its bytes are already inside that scenario's plan_mode_b
        # static hash below — this is the readable copy.
        "database_by_scenario": {
            s.id: {
                "path": (db := ctx.database_for(s)).path.name,
                "hash": db.db_hash,
                "program_name": next((r.get("name") for r in db.rows("programs")), None),
                "gen_ed_preference": s.gen_ed_preference,
                "world_language": (list(s.world_language) if s.world_language else None),
                # WHAT FRACTION OF THE FIXTURE'S OWN COURSE UNIVERSE this scenario was shown.
                # Mode B is scored against its own `database_for(s)` now (`real_scoring.py`), so
                # this is no longer a scorer-correctness signal — it is a "did
                # `real_db.program_id`/`--major` point at the program the fixture's scenarios
                # (student names, completed courses, credit targets) were actually written for"
                # signal. `report._guards` reads it back and flags a run where they diverge as
                # probably testing the wrong program, not as producing meaningless numbers.
                "fixture_overlap": (
                    round(len(db.course_codes & set(ctx.fixture.by_code))
                          / len(ctx.fixture.by_code), 3)
                    if ctx.fixture.by_code else None
                ),
                # Iterate `files`, not `tables`: `files` is keyed to exactly
                # `catalog_export.TABLES` (every table `render_context` prints), while
                # `tables` could in principle
                # hold something wider that never got hashed.
                "tables": {t: {"rows": len(db.rows(t)), "hash": h} for t, h in db.files.items()},
            }
            for s in ctx.fixture.scenarios
        },
        "static_hashes": {
            # Mode A/QA/Explain never read the database (see `PromptBuilder`), so one
            # representative scenario's hash stands in for all of them here — their prompts
            # still individually vary by scenario feedback/profile, exactly as before this
            # refactor; only Mode B's hash is genuinely per-scenario now.
            "plan_mode_a": ctx.prompts.plan_proposal(scenario, generate_plan(
                scenario.profile, ctx.fixture.catalog))[2],
            "plan_mode_b_by_scenario": {
                s.id: ctx.prompts_for(s).plan_freeform(s)[2] for s in ctx.fixture.scenarios
            },
            "qa": ctx.prompts.grounded_qa("x", [])[2],
            "explain": ctx.prompts.explain_plan("x", "{}")[2],
        },
    }
    (ctx.results_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    if not ctx.fixture.verified:
        print("[meta] NOTE: plan fixture is marked verified: false — rankings are usable, "
              "absolute degree-progress claims are not.")
