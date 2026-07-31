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
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import plan_scorers, scorers
from .fixtures import Fixture, SamplePlan, Scenario, load_fixture, load_sample_plan
from .llamacpp_client import GenerationResult, LlamaCppClient, LlamaCppError
from .mock_db import MockDatabase, load_mock_db
from .planner import (
    Plan, Profile, Proposal, apply_proposal, extract_credit_cap, generate_plan,
    set_planning_terms, severity,
)
from .prompts import (
    PROPOSAL_SCHEMA, PromptBuilder, catalog_tags, json_response_format, plan_schema,
    proposal_schema,
)
from .server import (
    ServerHandle, gpu_memory_used_mb, resolve_host, slot_context, start_server,
)

WARMUP_SYSTEM = "You reply with exactly one word."
WARMUP_USER = "Reply with the single word: ready"


@dataclass
class EvalContext:
    cfg: dict[str, Any]
    root: Path
    fixture: Fixture
    prompts: PromptBuilder
    questions: list[dict[str, Any]]
    host: str
    # The mock of the advisor's real databases — Mode B's entire context. Carried on the
    # context so its hash can travel into meta_*.json alongside the fixture's.
    database: MockDatabase
    # The published sample plan. NO LONGER A PROMPT INPUT — the reference arm was replaced by
    # the feedback arm on 2026-07-30. It is still loaded because `template_anchoring` scores
    # every free-form plan against it as a base rate: how much of the published template a
    # model reproduces from parametric memory alone. None simply omits that diagnostic.
    sample_plan: SamplePlan | None = None

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
        cap = profile.max_credits_per_semester
        soft = min(profile.preferred_major_courses_per_semester,
                   profile.max_major_courses_per_semester)
        hard = profile.max_major_courses_per_semester
        lines += [
            f"### Semesters (credit cap {cap}, {profile.major_subject} cap {hard}, "
            f"{soft} preferred)", "",
            f"| # | term | credits | {profile.major_subject} | courses |",
            "|---|---|---|---|---|",
        ]
        for i, credits in enumerate(score.semester_credits):
            major = (score.major_courses_per_semester[i]
                     if i < len(score.major_courses_per_semester) else 0)
            term, year = terms[i] if i < len(terms) else ("?", "?")
            codes = ", ".join(semesters[i]) if i < len(semesters) else ""
            over_cr = " **OVER**" if credits > cap else ""
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


def load_context(root: Path) -> EvalContext:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    # Do this BEFORE loading the fixture: the fixture's scenarios are planned with the
    # deterministic planner, which needs to already know whether summer is schedulable.
    set_planning_terms(cfg["run"]["planning_terms"])
    fixture = load_fixture(root / cfg["paths"]["plan_fixture"])
    questions_path = root / cfg["paths"]["questions"]
    questions = yaml.safe_load(questions_path.read_text()) if questions_path.exists() else []

    # THE MODEL'S CONTEXT. Loaded before the prompts are built, because Mode B's static block is
    # the rendered database — its bytes are in the static hash, so a stale mock db is a silently
    # different experiment, which is what `run.py check`'s staleness guard exists to prevent.
    database = load_mock_db(root / cfg["paths"]["mock_db"])

    sample_rel = cfg["paths"].get("sample_plan")
    sample_plan = load_sample_plan(root / sample_rel) if sample_rel else None
    if sample_rel and sample_plan is None:  # pragma: no cover — defensive
        print(f"[WARN] paths.sample_plan is set to {sample_rel} but nothing loaded; "
              "Mode C will be skipped.")

    return EvalContext(
        cfg=cfg,
        root=root,
        fixture=fixture,
        prompts=PromptBuilder(fixture, sample_plan, database),
        questions=questions or [],
        host=resolve_host(cfg["llamacpp"].get("host", "auto"),
                          cfg["llamacpp"].get("server_exe")),
        sample_plan=sample_plan,
        database=database,
    )


def sampling_options(cfg: dict[str, Any], *, max_tokens: int | None = None) -> dict[str, Any]:
    run = cfg["run"]
    return {
        "temperature": run["temperature"],
        "top_p": run.get("top_p", 0.9),
        "seed": run["seed"],
        "max_tokens": max_tokens or run["max_output_tokens"],
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
    return {
        "model": model_cfg["name"],
        "stage": "env",
        "gguf": model_cfg["gguf"],
        "bracket": model_cfg.get("bracket"),
        "think_setting": model_cfg.get("think"),
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
        "expected_slot_ctx": slot_context(ctx.cfg["run"]),
        "ctx_matches_config": (
            server_ctx == slot_context(ctx.cfg["run"]) if server_ctx else None),
        "loaded_model_id": client.loaded_model(),
        "build_info": props.get("build_info"),
    }


def _base_record(model: str, res: GenerationResult, static_hash: str, **extra) -> dict[str, Any]:
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
                options=sampling_options(ctx.cfg, max_tokens=ctx.cfg["run"]["max_output_tokens"]),
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
            str(best_proposal.get("rationale") or ""), planned_codes, ctx.fixture
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
    """The model emits the whole schedule. Scored directly against the fixture's rules.

    This is where models actually separate. A plan is viable only if it has ZERO hard
    violations and covers every requirement — one prerequisite mistake anywhere in eight
    semesters fails the whole plan, which is the correct standard: a student following it
    would be turned away at registration.
    """
    _run_freeform_plan(
        ctx, client, model_cfg, scenario, run_idx, out,
        mitigate=mitigate, stage="plan_mode_b",
        prompt=ctx.prompts.plan_freeform(scenario),
    )


def _run_freeform_plan(
    ctx: EvalContext, client: LlamaCppClient, model_cfg: dict,
    scenario: Scenario, run_idx: int, out, *, mitigate: bool,
    stage: str, prompt: tuple[str, str, str],
) -> None:
    system, user, static_hash = prompt
    try:
        res = client.chat(
            system, user,
            options=sampling_options(ctx.cfg, max_tokens=ctx.cfg["run"]["max_plan_tokens"]),
            think=model_cfg.get("think"),
            response_format=json_response_format(
                "PlanOfStudy", plan_schema(
                    ctx.cfg["run"]["planning_terms"],
                    [c.code for c in ctx.fixture.catalog],
                    [g["id"] for g in ctx.fixture.requirement_groups])),
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
        rec.update(plan_scorers.PlanScore().as_record())
        rec["structure_ok"] = False
        rec["parse_failure_reason"] = "no JSON object" if not parsed else "no semesters array"
        out.write(json.dumps(rec, default=str) + "\n")
        ctx.write_transcript(model_cfg["name"], stage, scenario.id, run_idx,
                             system, user, res.text,
                             verdict=f"UNPARSEABLE ({rec['parse_failure_reason']})")
        return

    semesters = [
        [str(c) for c in (s.get("courses") or [])]
        for s in parsed["semesters"] if isinstance(s, dict)
    ]
    score = plan_scorers.score_plan(
        ctx.fixture, scenario.profile, semesters, assertions=scenario.assertions
    )
    planned_codes = {plan_scorers.normalize_code(c) for s in semesters for c in s}
    rec.update(score.as_record())
    rec.update({
        "semesters": semesters,
        "semester_count": len(semesters),
        "over_horizon": len(semesters) > scenario.profile.semesters_to_plan,
        "declared_unplanned": parsed.get("unplanned") or [],
        # The model's own requirement accounting, stored raw. The derived flags below say
        # whether it agreed with the scorer; only the object itself says what it actually
        # claimed, and that is what you read when a flag looks wrong.
        "requirements_covered": parsed.get("requirements_covered"),
        "rationale": parsed.get("rationale"),
        "rationale_flags": plan_scorers.rationale_flags(
            str(parsed.get("rationale") or ""), planned_codes, ctx.fixture
        ),
        "term_labels_ok": _term_labels_ok(parsed["semesters"], scenario.profile),
        **plan_scorers.self_report_check(parsed["semesters"], score),
        **plan_scorers.requirement_report_check(
            parsed.get("requirements_covered"), ctx.fixture, scenario.profile,
            semesters, score),
    })
    # Anchoring is computed for BOTH modes whenever a sample plan is configured. Mode B never
    # sees it, so Mode B's slot_match is the base rate a model reproduces from memory alone —
    # without that baseline, Mode C's number is unreadable.
    if ctx.sample_plan is not None:
        rec.update(plan_scorers.template_anchoring(
            ctx.sample_plan, semesters, score.hallucinated,
            canonical=ctx.fixture.canonical))
    out.write(json.dumps(rec, default=str) + "\n")
    ctx.write_transcript(
        model_cfg["name"], stage, scenario.id, run_idx, system, user, res.text,
        verdict=f"viable={score.viable} coverage={score.requirement_coverage:.0%} "
                f"violations={score.violation_counts or 'none'} "
                f"slot_match={rec.get('template_slot_match')}",
        detail=_score_detail(score, semesters, scenario.profile),
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
            options=sampling_options(ctx.cfg),
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
    # ``PlannedCourse`` objects (code + title + credits + workload), NOT bare course codes.
    # Sending codes alone was a harness-only bug with a measurable cost: with no title in the
    # payload, models fill the gap from parametric memory and tell the student CS 35400 is
    # "Theory of Computation" (it is Operating Systems). That failure mode is an artifact of
    # the harness, not something production can produce, so scoring it was scoring a fiction.
    by_code = {c.code: c for c in ctx.fixture.catalog}
    def _planned_course(code: str) -> dict:
        course = by_code.get(code)
        if course is None:  # a plan can only hold catalog codes; be explicit if that breaks
            return {"code": code, "title": "", "credits": 0, "workload_score": 0}
        return {
            "code": course.code, "title": course.title,
            "credits": course.credits, "workload_score": course.workload_score,
        }

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
            system, user, options=sampling_options(ctx.cfg), think=model_cfg.get("think")
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
    rec["needs_review"] = True
    out.write(json.dumps(rec, default=str) + "\n")
    ctx.write_transcript(
        model_cfg["name"], "explain", scenario.id, run_idx, system, user, res.text,
        verdict=f"faith_flags={len(rec['faithfulness_flags'])} truncated={res.truncated}",
    )


# --- top level --------------------------------------------------------------------------------


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

        runs = ctx.cfg["run"]["runs_per_pair"]
        for run_idx in range(runs):
            # Plan first: the feature that matters most gets the data if a run is cut short.
            for scenario in ctx.fixture.scenarios:
                if "plan_a" in tasks:
                    run_mode_a(ctx, client, model_cfg, scenario, run_idx, out, mitigate=mitigate)
                if "plan_b" in tasks:
                    run_mode_b(ctx, client, model_cfg, scenario, run_idx, out, mitigate=mitigate)
                if "explain" in tasks:
                    run_explain(ctx, client, model_cfg, scenario, run_idx, out, mitigate=mitigate)
                out.flush()
            if "qa" in tasks:
                for question in ctx.questions:
                    run_qa(ctx, client, model_cfg, question, run_idx, out, mitigate=mitigate)
                out.flush()
            print(f"  [{name}] replicate {run_idx + 1}/{runs} done")
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


def write_meta(ctx: EvalContext, tag: str, selected: list[dict], tasks: set[str]) -> None:
    scenario = ctx.fixture.scenarios[0]
    meta = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": ctx.host,
        "tasks": sorted(tasks),
        "models": [m["name"] for m in selected],
        "run": ctx.cfg["run"],
        "llamacpp": ctx.cfg["llamacpp"],
        "fixture": {
            "path": str(ctx.fixture.path.name),
            "hash": ctx.fixture.fixture_hash,
            "verified": ctx.fixture.verified,
            "scenarios": [s.id for s in ctx.fixture.scenarios],
            "courses": len(ctx.fixture.catalog),
            "requirement_groups": len(ctx.fixture.requirement_groups),
        },
        # Mode B's whole context. Recorded per table, not just as one hash: when the export
        # changes, "which table moved" is the first question, and a single digest cannot answer
        # it. Its bytes are already inside plan_mode_b's static hash — this is the readable copy.
        "mock_db": {
            "path": ctx.database.path.name,
            "hash": ctx.database.db_hash,
            "tables": {t: {"rows": len(ctx.database.rows(t)), "hash": ctx.database.files[t]}
                       for t in ctx.database.tables},
        },
        "static_hashes": {
            "plan_mode_a": ctx.prompts.plan_proposal(scenario, generate_plan(
                scenario.profile, ctx.fixture.catalog))[2],
            "plan_mode_b": ctx.prompts.plan_freeform(scenario)[2],
            "qa": ctx.prompts.grounded_qa("x", [])[2],
            "explain": ctx.prompts.explain_plan("x", "{}")[2],
        },
    }
    if ctx.sample_plan is not None:
        # Recorded separately from fixture_hash on purpose: the sample plan is prompt input,
        # not scoring authority. It cannot change a Mode A/B score, and folding it into
        # fixture_hash would invalidate those records for a change they never saw.
        meta["sample_plan"] = {
            "path": ctx.sample_plan.path.name,
            "hash": ctx.sample_plan.plan_hash,
            "url": ctx.sample_plan.source.get("url"),
            "semesters": len(ctx.sample_plan.semesters),
            "named_courses": len(ctx.sample_plan.slot_of),
            "off_catalog_codes": sorted(ctx.sample_plan.off_catalog_codes),
        }
    (ctx.results_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    if not ctx.fixture.verified:
        print("[meta] NOTE: plan fixture is marked verified: false — rankings are usable, "
              "absolute degree-progress claims are not.")
