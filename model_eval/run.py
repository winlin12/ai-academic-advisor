#!/usr/bin/env python3
"""Evaluation harness CLI. Standalone: Python 3.10+ and PyYAML, nothing from the app.

  python run.py doctor                      diagnose local llama-server reachability
  python run.py check                       validate fixture, prompts, GPU, model files
  python run.py serve <model>               launch llama-server for one model and hold it
  python run.py run [--models a,b] [--brackets 8gb,coder] [--tasks plan_a,plan_b,qa]
  python run.py run --mitigate --models X   mitigation pass (multi-iteration revise loop)
  python run.py converge --preflight        MODE C: validate locked slots, spend no GPU time
  python run.py converge [--variants blind,feedback] [--models a,b] [--scenarios id]
  python run.py converge-report             results/convergence_report.md
  python run.py report                      plan tables + validity guards + review queue
  python run.py parity                      check the vendored planner against the app's
  python run.py fixture-check               diff the plan fixture against the catalog DB

The harness starts and stops llama-server itself (see harness/server.py): under llama.cpp,
context size and GPU/KV settings are launch flags, so "every model saw identical settings"
is only true if one place owns the command line.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="diagnose local llama-server setup")
    sub.add_parser("check", help="validate fixture, prompts, GPU, and model files")
    sub.add_parser("report", help="generate results/report.md")
    sub.add_parser("parity", help="check the vendored planner against the app's")
    sub.add_parser("fixture-check", help="diff the plan fixture against the catalog DB")
    sub.add_parser("refresh-offerings",
                   help="rewrite the fixture's offered_terms from observed PurdueIO offerings")
    sub.add_parser("build-mock-db",
                   help="regenerate mock_db/ from the plan fixture (run after editing it)")

    servep = sub.add_parser("serve", help="launch llama-server for one model and hold it")
    servep.add_argument("model", help="model name from config.yaml")

    convp = sub.add_parser(
        "converge", help="MODE C: retry-to-convergence (attempts + wall clock, censored)")
    convp.add_argument("--models", help="comma-separated model names (default: all)")
    convp.add_argument("--brackets", help="comma-separated brackets: 8gb,coder,server,reference")
    convp.add_argument("--variants", help="comma-separated: blind,feedback (default: both)")
    convp.add_argument("--scenarios", help="comma-separated scenario ids (default: all)")
    convp.add_argument("--preflight", action="store_true",
                       help="validate locked slots and print the prereq-risk note, then stop")

    sub.add_parser("converge-report", help="results/convergence_report.md")

    runp = sub.add_parser("run", help="execute the evaluation")
    runp.add_argument("--models", help="comma-separated model names (default: all)")
    runp.add_argument("--brackets", help="comma-separated brackets: 8gb,coder,server,reference")
    runp.add_argument("--tasks", help="comma-separated: plan_a,plan_b,qa,explain")
    runp.add_argument("--mitigate", action="store_true",
                      help="mitigation mode: multi-iteration revise loop; writes "
                           "runs_mitigated.jsonl")

    args = ap.parse_args()

    if args.cmd == "doctor":
        doctor()
    elif args.cmd == "check":
        check()
    elif args.cmd == "serve":
        serve(args.model)
    elif args.cmd == "run":
        from harness.runner import ConcurrentRunError, run_eval
        try:
            _run(args, run_eval)
        except ConcurrentRunError as exc:
            # A guard, not a crash — print the reason and the fix, not a traceback.
            print(f"refusing to start: {exc}")
            sys.exit(1)
    elif args.cmd == "converge":
        converge(args)
    elif args.cmd == "converge-report":
        from harness.convergence_report import generate_report as convergence_report
        print(f"wrote {convergence_report(ROOT)}")
    elif args.cmd == "report":
        from harness.report import generate_report
        generate_report(ROOT)
    elif args.cmd == "parity":
        parity()
    elif args.cmd == "fixture-check":
        fixture_check()
    elif args.cmd == "refresh-offerings":
        refresh_offerings()
    elif args.cmd == "build-mock-db":
        build_mock_db()


def _run(args, run_eval) -> None:
    run_eval(
        ROOT,
        models=args.models.split(",") if args.models else None,
        brackets=args.brackets.split(",") if args.brackets else None,
        tasks=args.tasks.split(",") if args.tasks else None,
        mitigate=args.mitigate,
    )


def _cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# --- converge (MODE C) --------------------------------------------------------------------------


def converge(args) -> None:
    """Mode C. ``--preflight`` spends no GPU time; it is what you run after editing locks."""
    from harness.convergence import (
        load_locked_slots, preflight, prereq_risk_note, run_convergence,
    )
    from harness.runner import ConcurrentRunError

    split = lambda v: v.split(",") if v else None

    if args.preflight:
        from harness.runner import load_context
        ctx = load_context(ROOT)
        conv_cfg = ctx.cfg.get("convergence") or {}
        if not conv_cfg:
            print("config.yaml has no `convergence:` block.")
            sys.exit(1)
        locked = load_locked_slots(ROOT / conv_cfg["locked_slots"])
        print(prereq_risk_note(ctx.fixture, conv_cfg))
        print(f"\nlocked slots: {locked.path.name} (hash {locked.slots_hash})")
        for scenario in ctx.fixture.scenarios:
            slots = locked.for_scenario(scenario.id)
            pins = ", ".join(f"{s.course}@{s.semester_index + 1}" for s in slots) or "none"
            print(f"  {scenario.id:24s} {pins}")
        problems = preflight(ctx, locked, conv_cfg)
        if problems:
            print(f"\n❌ {len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")
            sys.exit(1)
        print("\n✅ preflight passed: every locked slot is legal where it is pinned.")
        return

    try:
        run_convergence(
            ROOT,
            models=split(args.models),
            brackets=split(args.brackets),
            variants=split(args.variants),
            scenarios=split(args.scenarios),
        )
    except ConcurrentRunError as exc:
        print(f"refusing to start: {exc}")
        sys.exit(1)


# --- build-mock-db -----------------------------------------------------------------------------


def build_mock_db() -> None:
    """Regenerate mock_db/ from the plan fixture.

    The mock database is what the model is shown; the fixture is what the model is scored
    against. Generating one from the other is what keeps those two the same set of facts —
    hand-maintaining both is how a model ends up penalised for a prerequisite it was never
    given. `run.py check` fails if this has not been run since the fixture last changed.
    """
    from harness.fixtures import load_fixture
    from harness.mock_db import load_mock_db, integrity_problems, render_context
    from harness.mock_db_build import write_mock_db
    from harness.server import approx_tokens

    cfg = _cfg()
    fixture = load_fixture(ROOT / cfg["paths"]["plan_fixture"])
    out_dir = ROOT / cfg["paths"]["mock_db"]
    written = write_mock_db(fixture, out_dir)
    print(f"wrote {len(written)} table(s) to {out_dir}")

    db = load_mock_db(out_dir)
    for table in db.tables:
        print(f"  {table:28s} {len(db.rows(table)):4d} rows   (sha {db.files[table]})")
    problems = integrity_problems(db)
    if problems:
        print(f"\n❌ {len(problems)} integrity problem(s) in the generated database:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    export = render_context(db)
    print(f"\ndb_hash {db.db_hash} | export {len(export)} chars "
          f"(~{approx_tokens(export)} tokens)")


# --- doctor ------------------------------------------------------------------------------------


def doctor() -> None:
    """The most likely reasons a run fails on this box, diagnosed explicitly.

    llama-server now runs as a native Linux process in this same WSL box (built from
    ./llama.cpp), so there is no cross-VM networking to diagnose — just: does the binary
    exist, is a GPU visible, and is the configured port free (or already serving what you
    expect).
    """
    from harness.server import gpu_memory_used_mb, resolve_host

    cfg = _cfg()
    port = cfg["llamacpp"]["port"]
    host = resolve_host(cfg["llamacpp"].get("host", "auto"),
                        cfg["llamacpp"].get("server_exe"))
    print("=== local llama-server setup ===\n")

    exe = Path(cfg["llamacpp"]["server_exe"])
    print(f"llama-server: {'found' if exe.exists() else 'NOT FOUND at ' + str(exe)}")
    if not exe.exists():
        print("  Build it: cd llama.cpp && cmake -B build -DGGML_CUDA=on && "
              "cmake --build build --config Release -j")

    models_root = Path(cfg["llamacpp"]["models_root"])
    print(f"models_root: {'found' if models_root.exists() else 'NOT FOUND at ' + str(models_root)}")

    gpu = gpu_memory_used_mb()
    print(f"nvidia-smi: {'available, ' + str(gpu) + ' MB used' if gpu else 'NOT available'}")

    listening = _port_open(host, port, timeout=2)
    print(f"\n{host}:{port} -> {'OPEN' if listening else 'closed'}")

    if listening:
        print("\nA server is already reachable on that port. If it is not the model you want to "
              "test, stop it — `run.py run` will start its own.")
    else:
        print("\nNo server is listening there right now, which is expected when one isn't "
              "running. `python run.py serve <model>` or `python run.py run` will start one.")


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


# --- check -------------------------------------------------------------------------------------


def check() -> None:
    """Everything verifiable without spending a single generation."""
    from harness.fixtures import load_fixture, load_sample_plan
    from harness.mock_db import integrity_problems, load_mock_db
    from harness.mock_db_build import stale_tables
    from harness.planner import generate_plan
    from harness.plan_scorers import score_plan, plan_to_semesters
    from harness.prompts import PromptBuilder
    from harness.server import gpu_memory_used_mb

    cfg = _cfg()
    fixture = load_fixture(ROOT / cfg["paths"]["plan_fixture"])
    sample_rel = cfg["paths"].get("sample_plan")
    sample_plan = load_sample_plan(ROOT / sample_rel) if sample_rel else None
    mock_dir = ROOT / cfg["paths"]["mock_db"]
    database = load_mock_db(mock_dir)
    prompts = PromptBuilder(fixture, sample_plan, database)

    print(f"fixture: {fixture.name}")
    print(f"  hash {fixture.fixture_hash} | {len(fixture.catalog)} courses | "
          f"{len(fixture.requirement_groups)} requirement groups | "
          f"{len(fixture.scenarios)} scenarios")
    if not fixture.verified:
        print("  ⚠️  verified: false — PREREQ EDGES are still hand-written (Purdue publishes "
              "them only in Banner, whose robots.txt disallows crawling). Term offerings are "
              "now observed from PurdueIO; re-run `run.py refresh-offerings` after a sync. "
              "Rankings are usable; absolute degree claims are not.")

    # Internal consistency: does every code referenced anywhere actually exist?
    known = set(fixture.by_code)
    problems = []
    for course in fixture.catalog:
        for prereq in course.prereqs:
            if prereq not in known:
                problems.append(f"{course.code} lists unknown prereq {prereq}")
        for coreq in course.coreqs:
            if coreq not in known:
                problems.append(f"{course.code} lists unknown coreq {coreq}")
            if coreq in course.prereqs:
                problems.append(
                    f"{course.code} lists {coreq} as BOTH a prereq and a coreq — the prereq "
                    f"reading wins in the scorer, so the coreq is silently a no-op."
                )
        if not course.offered_terms:
            problems.append(f"{course.code} has no offered_terms")
    for group in fixture.requirement_groups:
        for code in group.get("courses", []):
            if code not in known:
                problems.append(f"requirement group {group['id']} lists unknown course {code}")

    # `equivalent_to` is load-bearing for three separate checks (duplicates, prereqs,
    # coverage), and every way it can be wrong is silent at runtime.
    grouped = {c for g in fixture.requirement_groups for c in g.get("courses", [])}
    for course in fixture.catalog:
        if not course.equivalent_to:
            continue
        if course.equivalent_to not in known:
            problems.append(
                f"{course.code} is equivalent_to {course.equivalent_to}, which is not in the "
                f"catalog — the substitution would silently never apply."
            )
        if course.code in grouped:
            problems.append(
                f"{course.code} is a substitute for {course.equivalent_to} but is ALSO listed "
                f"in a requirement group. It would be counted twice toward a `choose` target "
                f"and the deterministic planner would schedule both. List only the primary."
            )
        if fixture.canonical.get(course.code) == course.code:
            problems.append(
                f"{course.code}'s equivalence resolves back to itself — a cycle in "
                f"`equivalent_to`. Substitution is silently a no-op for it."
            )
    for scenario in fixture.scenarios:
        for code in scenario.profile.completed_courses:
            if code not in known:
                problems.append(f"scenario {scenario.id} completed unknown course {code}")

    # The Mode C reference plan. It scores nothing, so a mistake in it cannot make a plan
    # wrongly viable — but a `code:` that the catalog does not have would silently drop out of
    # the anchoring slot map, quietly shrinking the denominator of the one number this arm
    # turns on. That is worth a hard failure; an `also_cites` code NOT in the catalog is the
    # normal case and is exactly what makes the copied-hallucination diagnostic work.
    if sample_plan is not None:
        print(f"\nSample plan (anchoring diagnostic only, no longer a prompt input): "
              f"{sample_plan.path.name} (hash {sample_plan.plan_hash})")
        rows = sum(len(s.entries) for s in sample_plan.semesters)
        named = sample_plan.slot_of
        print(f"  {len(sample_plan.semesters)} semesters | {rows} rows | "
              f"{len(named)} name a catalog course | {rows - len(named)} placeholders")
        print(f"  off-catalog codes it cites (alternatives/suggestions, kept on purpose): "
              f"{', '.join(sorted(sample_plan.off_catalog_codes)) or 'none'}")
        for semester in sample_plan.semesters:
            for entry in semester.entries:
                if entry.code and entry.code not in known:
                    problems.append(
                        f"sample plan row {entry.code!r} ({semester.label}) sets `code:` to a "
                        f"course the scoring fixture does not have — move it to `also_cites` "
                        f"or add the course, or it silently drops out of the anchoring metric."
                    )
                for cited in entry.also_cites:
                    if cited in known:
                        problems.append(
                            f"sample plan row cites {cited} under `also_cites` but the fixture "
                            f"DOES have it — a model scheduling it would be scored legal while "
                            f"the diagnostic calls it an off-catalog copy."
                        )
        # Not fatal: the published plan genuinely omits courses the fixture requires (CS 38100),
        # and that gap is part of what Mode C tests. Printing it stops it being read as a bug.
        required = {c for g in fixture.requirement_groups if g.get("kind") == "all"
                    for c in g.get("courses", [])}
        uncovered = sorted(required - set(named))
        if uncovered:
            print(f"  NOTE: required courses the sample plan never names: "
                  f"{', '.join(uncovered)} — following it literally does not complete the "
                  f"degree, which the model is expected to notice.")
    elif sample_rel:
        problems.append(f"paths.sample_plan points at {sample_rel} but it could not be loaded.")
    else:
        print("\nSample plan: not configured (paths.sample_plan unset) — the template-anchoring "
              "diagnostic will be absent. Mode C does not need it.")

    # THE check that matters: is a viable plan even reachable? If the deterministic planner —
    # which cannot make a mistake — can't produce a viable plan for a scenario, then every
    # model scores 0 on it for reasons that have nothing to do with the model.
    print("\nreachability (deterministic planner vs. the scoring rules):")
    for scenario in fixture.scenarios:
        plan = generate_plan(scenario.profile, fixture.catalog)
        score = score_plan(fixture, scenario.profile, plan_to_semesters(plan))
        status = "VIABLE" if score.viable else "NOT VIABLE"
        suffix = " (expected — scenario is unsatisfiable by design)" \
            if scenario.expect_unsatisfiable else ""
        print(f"  {scenario.id:24s} {status:11s} coverage={score.requirement_coverage:.0%} "
              f"violations={len(score.violations)}{suffix}")
        if not score.viable and not scenario.expect_unsatisfiable:
            problems.append(
                f"scenario {scenario.id}: the deterministic planner itself cannot produce a "
                f"viable plan — so no model can, and every model scores 0 here for reasons "
                f"unrelated to the model. Missing: "
                f"{'; '.join(score.missing_requirements[:3]) or score.violations[:3]}"
            )
        if score.viable and scenario.expect_unsatisfiable:
            problems.append(
                f"scenario {scenario.id} is marked expect_unsatisfiable but the planner "
                f"solved it — the scenario has gone soft and no longer tests honest "
                f"reporting of what doesn't fit."
            )

    # THE MOCK DATABASE — what Mode B is actually shown. Two failures are checked, and both are
    # silent otherwise: rows that reference something that does not exist (a model scored
    # against a requirement whose course it never saw), and a database that no longer matches
    # the fixture it was generated from (a model shown one catalog and scored against another).
    print(f"\nmock database: {mock_dir.name} (hash {database.db_hash})")
    print("  " + " | ".join(f"{t} {len(database.rows(t))}" for t in database.tables))
    problems += [f"mock db: {p}" for p in integrity_problems(database)]
    stale = stale_tables(fixture, mock_dir)
    if stale:
        problems.append(
            f"mock database is STALE — {', '.join(stale)} would be regenerated differently from "
            f"the current fixture. The model would be shown one catalog and scored against "
            f"another. Run `python run.py build-mock-db`."
        )
    else:
        print("  in sync with the fixture ✓")

    questions = _load_questions(cfg)
    behaviors = {}
    for q in questions:
        behaviors[q["expected_behavior"]] = behaviors.get(q["expected_behavior"], 0) + 1
    print(f"\nquestions: {len(questions)} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(behaviors.items()))})")

    scenario = fixture.scenarios[0]
    plan = generate_plan(scenario.profile, fixture.catalog)
    print("\nstatic prompt hashes:")
    print(f"  plan_mode_a {prompts.plan_proposal(scenario, plan)[2]}")
    print(f"  plan_mode_b {prompts.plan_freeform(scenario)[2]}")
    print(f"  qa          {prompts.grounded_qa('x', [])[2]}")
    print(f"  explain     {prompts.explain_plan('x', '{}')[2]}")

    # DOES THE PROMPT FIT? Not a model failure, so it is caught here rather than scored. Mode B
    # is the long one — it carries the whole database export — and it is what decides whether
    # num_ctx is big enough. This guard is the reason a `num_ctx: 8192` edit fails `check`
    # instead of producing 135 HTTP 400s halfway through a sweep.
    from harness.server import (  # noqa: PLC0415
        approx_tokens, check_slot_context, slot_context,
    )
    budget = cfg["run"]["num_ctx"] - cfg["run"]["max_plan_tokens"]
    slot = slot_context(cfg["run"])
    print(f"\nper-slot context: {slot} tokens "
          f"(num_ctx {cfg['run']['num_ctx']} / parallel {cfg['run'].get('parallel', 1)}), "
          f"minus max_plan_tokens {cfg['run']['max_plan_tokens']} = "
          f"{slot - cfg['run']['max_plan_tokens']} left for the prompt")
    system, user, _ = prompts.plan_freeform(scenario)
    approx = approx_tokens(system + user)
    print(f"  mode B: ~{approx} tokens")
    if approx > budget:
        problems.append(
            f"the mode B prompt (~{approx} tok) does not leave room to generate within "
            f"num_ctx={cfg['run']['num_ctx']}. Raise num_ctx, or shrink the database export."
        )
    # Does ONE request still fit? `--parallel` divides the context, so this catches
    # "parallel 4 quietly made every plan request truncate against the context".
    slot_problem = check_slot_context(cfg["run"], approx)
    if slot_problem:
        problems.append(slot_problem)

    # Two entries pointing at the same gguf benchmark ONE model under two names and report a
    # fake head-to-head. The report's duplicate-NAME guard cannot see this: the names differ.
    # It is an easy copy-paste mistake — and it has already happened once in this file.
    by_gguf: dict[str, list[str]] = {}
    for model in cfg["models"]:
        by_gguf.setdefault(model["gguf"], []).append(model["name"])
    for gguf, names in by_gguf.items():
        if len(names) > 1:
            problems.append(
                f"models {', '.join(names)} all point at the same gguf ({gguf}) — they would "
                f"benchmark one model under several names and fabricate a head-to-head."
            )

    print("\nmodel files:")
    root = Path(cfg["llamacpp"]["models_root"])
    missing = 0
    for model in cfg["models"]:
        path = root / model["gguf"]
        if not path.exists():
            missing += 1
            print(f"  MISSING  {model['name']}: {path}")
    print(f"  {len(cfg['models']) - missing}/{len(cfg['models'])} gguf files present")

    gpu = gpu_memory_used_mb()
    print(f"\nnvidia-smi: {'available, ' + str(gpu) + ' MB used' if gpu else 'NOT available'}")

    if problems:
        print(f"\n❌ {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\n✅ check passed. Run `python run.py doctor` before the first run on a new box.")


def _load_questions(cfg: dict) -> list:
    import yaml
    path = ROOT / cfg["paths"]["questions"]
    return yaml.safe_load(path.read_text()) if path.exists() else []


# --- serve -------------------------------------------------------------------------------------


def serve(model_name: str) -> None:
    """Launch llama-server for one model with the harness's exact settings and hold it.

    Useful for eyeballing a model by hand, or pointing the real backend at an eval model
    without hand-copying flags out of config.yaml.
    """
    from harness.server import start_server

    cfg = _cfg()
    matches = [m for m in cfg["models"] if m["name"] == model_name]
    if not matches:
        print(f"Unknown model {model_name!r}. Known: "
              f"{', '.join(m['name'] for m in cfg['models'])}")
        sys.exit(1)

    handle = start_server(cfg, matches[0], ROOT / "results" / "server_logs")
    print(f"llama-server up: {handle.base_url}")
    print(f"  argv: {' '.join(handle.argv)}")
    print(f"  log:  {handle.log_path}")
    print(f"  offload: {handle.offload()}")
    print("\nCtrl-C to stop.")
    try:
        handle.process.wait()
    except KeyboardInterrupt:
        handle.stop()
        print("\nstopped.")


# --- parity ------------------------------------------------------------------------------------


def parity() -> None:
    """Is harness/planner.py still equivalent to the app's planner?

    Mode A only measures production if these agree. This is a best-effort check: it needs the
    backend importable (pydantic installed), which the harness deliberately does not require,
    so it degrades to a clear "could not verify" rather than a false pass.
    """
    from harness.fixtures import load_fixture
    from harness import planner as vendored

    backend = ROOT.parent / "backend"
    sys.path.insert(0, str(backend))
    try:
        from app.models.schemas import Course as AppCourse  # noqa: PLC0415
        from app.models.schemas import StudentProfile as AppProfile  # noqa: PLC0415
        from app.services.planner import generate_plan as app_generate  # noqa: PLC0415
    except ImportError as exc:
        print(f"⚠️  Could not import the app's planner ({exc}).")
        print("    Install the backend's deps and re-run, or accept that Mode A's claim to")
        print("    measure production is UNVERIFIED in this report.")
        sys.exit(2)

    cfg = _cfg()
    fixture = load_fixture(ROOT / cfg["paths"]["plan_fixture"])
    # EVERY field the planner reads has to cross this bridge. Anything omitted silently gives
    # the app's planner different inputs, and the divergence then reads as "the port drifted"
    # when it is really the harness lying to itself — `coreqs` did exactly that on 2026-07-30.
    app_catalog = [
        AppCourse(
            code=c.code, title=c.title, credits=c.credits, prereqs=list(c.prereqs),
            coreqs=list(c.coreqs),
            offered_terms=list(c.offered_terms), requirement_tags=list(c.requirement_tags),
            workload_score=c.workload_score,
        )
        for c in fixture.catalog
    ]

    mismatches = 0
    for scenario in fixture.scenarios:
        p = scenario.profile
        mine = vendored.generate_plan(p, fixture.catalog)
        theirs = app_generate(
            AppProfile(
                name=p.name, degree_program=p.degree_program,
                completed_courses=list(p.completed_courses),
                remaining_courses=list(p.remaining_courses),
                start_term=p.start_term, start_year=p.start_year,
                semesters_to_plan=p.semesters_to_plan,
                max_credits_per_semester=p.max_credits_per_semester,
                target_credits_per_semester=p.target_credits_per_semester,
                major_subject=p.major_subject,
                max_major_courses_per_semester=p.max_major_courses_per_semester,
                preferred_major_courses_per_semester=p.preferred_major_courses_per_semester,
            ),
            app_catalog,
        )
        mine_sem = [s.courses for s in mine.semesters]
        theirs_sem = [[c.code for c in s.courses] for s in theirs.semesters]
        if mine_sem != theirs_sem or mine.unplanned_courses != theirs.unplanned_courses:
            mismatches += 1
            print(f"❌ {scenario.id} DIVERGED")
            print(f"   harness: {mine_sem}")
            print(f"   app:     {theirs_sem}")
        else:
            print(f"✅ {scenario.id}")

    if mismatches:
        print(f"\n{mismatches} scenario(s) diverged — harness/planner.py has drifted from "
              f"backend/app/services/planner.py. Mode A is NOT measuring production until "
              f"this is fixed.")
        sys.exit(1)
    print("\n✅ vendored planner matches the app's on every scenario.")


# --- fixture-check ----------------------------------------------------------------------------


def fixture_check() -> None:
    """Diff the plan fixture's course rows against a populated catalog database.

    Verifies codes, titles and credits. It CANNOT verify prereqs or offered_terms — Purdue's
    Acalog catalog publishes neither (TODO.md §2.5), which is exactly why those fields are
    hand-written in the fixture and flagged as the highest-risk rows in it.
    """
    from harness.fixtures import load_fixture

    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        print("psycopg is not installed (the harness does not require it). "
              "`pip install psycopg[binary]` to run this check.")
        sys.exit(2)

    cfg = _cfg()
    fixture = load_fixture(ROOT / cfg["paths"]["plan_fixture"])
    dsn = cfg.get("catalog_database_url") or \
        "postgresql://catalog:catalog@localhost:5433/catalog_ingestion"

    try:
        conn = psycopg.connect(dsn, connect_timeout=5)
    except psycopg.Error as exc:
        print(f"Cannot reach the catalog database at {dsn}: {exc}")
        print("Start it with `cd ../catalog_ingestion && docker compose up -d postgres`, and "
              "make sure it has been populated (`make restore` or `make sync-programs`).")
        sys.exit(2)

    codes = sorted(fixture.by_code)
    with conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM courses")
        (total,) = cur.fetchone()
        if not total:
            print("The catalog database is EMPTY — nothing to verify against. "
                  "Restore a backup or re-run the crawl first.")
            sys.exit(2)
        cur.execute(
            "SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min "
            "FROM courses WHERE course_code = ANY(%s) ORDER BY course_code", (codes,)
        )
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    print(f"catalog has {total} courses; checking {len(codes)} fixture codes\n")
    issues = 0
    for code in codes:
        course = fixture.by_code[code]
        if code not in rows:
            issues += 1
            print(f"  NOT IN CATALOG  {code} \"{course.title}\"")
            continue
        title, credits = rows[code]
        if credits is not None and int(round(float(credits))) != course.credits:
            issues += 1
            print(f"  CREDITS         {code}: fixture {course.credits} vs catalog {credits}")
        if title and title.strip().lower() != course.title.strip().lower():
            print(f"  title differs   {code}: {course.title!r} vs {title.strip()!r}")

    print(f"\n{issues} hard issue(s) found across {len(codes)} courses.")
    print("NOT CHECKED: prereqs and offered_terms — the catalog does not publish either. "
          "Those still need the course scheduler or the department's plan of study.")
    if issues == 0:
        print("\nIf this run was clean, flip `verified: true` in the fixture's `program:` "
              "block for the codes/credits half, and leave a note about the prereq half.")




# --- refresh-offerings -------------------------------------------------------------------------


PURDUEIO_DSN = "postgresql://purdueio:changeme_in_production@172.18.0.4:5432/purdueio"


def refresh_offerings() -> None:
    """Replace the fixture's hand-written ``offered_terms`` with observed PurdueIO data.

    This is the single highest-value correction available to the harness. Term offerings were
    invented (Purdue's Acalog catalog publishes none), and they were invented WRONG: CS 47300
    was recorded as spring-only when it is observed fall-only, CS 47100 as fall-only when it
    runs both — so models were being charged term-offering violations for schedules that were
    in fact legal, and credited for ones that were not.

    Offerings here are OBSERVED, not declared: PurdueIO has no "offered in" field, only
    ``Classes`` rows that actually ran. The observation count travels into the fixture as a
    comment so a one-sighting inference is visibly weaker than a nine-sighting one.

    Rewrites the YAML in place, line-wise, so every comment and the provenance header survive.
    """
    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        print("psycopg is not installed (the harness core does not need it). "
              "`pip install psycopg[binary]` to run this.")
        sys.exit(2)

    cfg = _cfg()
    fixture_path = ROOT / cfg["paths"]["plan_fixture"]
    dsn = cfg.get("purdueio_database_url", PURDUEIO_DSN)

    import re

    import yaml
    codes = [c["code"] for c in yaml.safe_load(fixture_path.read_text())["courses"]]

    try:
        with psycopg.connect(dsn, connect_timeout=8) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT course_code, offered_terms, terms_observed "
                "FROM advisor.course_planner_terms WHERE course_code = ANY(%s)", (codes,)
            )
            observed = {r[0]: (list(r[1]), r[2]) for r in cur.fetchall()}
    except psycopg.Error as exc:
        print(f"Cannot reach the purdueio database at {dsn}: {exc}")
        print("Start it, then run: cd ../backend && python -m app.services.offerings.sync")
        sys.exit(2)

    missing = [c for c in codes if c not in observed]
    lines = fixture_path.read_text().splitlines()
    out: list[str] = []
    current: str | None = None
    changed = unchanged = 0

    for line in lines:
        code_match = re.match(r'^  - code: "([^"]+)"', line)
        if code_match:
            current = code_match.group(1)
        term_match = re.match(r"^(\s*)offered_terms: \[(.*)\]\s*$", line)
        if term_match and current in observed:
            terms, n = observed[current]
            was = [t.strip() for t in term_match.group(2).split(",") if t.strip()]
            if sorted(was) != sorted(terms):
                changed += 1
                print(f"  {current:<12} {was} -> {terms}  ({n} terms observed)")
            else:
                unchanged += 1
            out.append(
                f"{term_match.group(1)}offered_terms: [{', '.join(terms)}]"
                f"   # observed in {n} PurdueIO terms"
            )
            continue
        out.append(line)

    # The header's provenance claim is now false for this field, and a stale caveat is worse
    # than none: it trains the reader to ignore the ones that are still true.
    text = "\n".join(out) + "\n"
    text = text.replace(
        "#   offered_terms ..................... ALSO NOT PUBLISHED. Hand-written. The",
        "#   offered_terms ..................... OBSERVED from PurdueIO Classes rows via\n"
        "#                                        `run.py refresh-offerings` — a course is\n"
        "#                                        'offered in fall' because it RAN in falls,\n"
        "#                                        not because anything declares it. The count\n"
        "#                                        in each row's comment is the evidence: 9\n"
        "#                                        terms is a pattern, 3 is a hint, and absence\n"
        "#                                        is the weakest signal of all. Previously\n"
        "#                                        hand-written, and wrong (CS 47300 was\n"
        "#                                        recorded spring-only; it runs fall-only).\n"
        "#   (superseded note) ................. The",
    )
    fixture_path.write_text(text)

    print(f"\n{changed} course(s) corrected, {unchanged} already matched.")
    if missing:
        print(f"{len(missing)} not in the offerings data (never observed in the synced terms): "
              f"{', '.join(missing)}")
        print("Those keep their hand-written terms — check whether the course still exists.")
    print("\nNOTE: prereq edges in the fixture are still hand-written. Purdue publishes them "
          "only in Banner, whose robots.txt disallows crawling.")


if __name__ == "__main__":
    main()
