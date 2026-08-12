#!/usr/bin/env python3
"""Evaluation harness CLI. Standalone: Python 3.10+ and PyYAML, nothing from the app.

  python run.py doctor                      diagnose local llama-server reachability
  python run.py check                       validate fixture, prompts, GPU, model files
  python run.py check --major "nursing"        same, but against a different school's plan
                                             fixture — matches plan_fixtures/*.yaml by slug,
                                             filename, or program name (also works on `run`)
  python run.py serve <model>               launch llama-server for one model and hold it
  python run.py run                         EVERY SCHOOL in plan_fixtures/, one after another,
                                             each into its own results/results_<slug>/ — see
                                             run.default_tasks for the task set each one runs
  python run.py run --major cs              just one school (see `check --major` above)
  python run.py run [--models a,b] [--brackets 8gb,coder] [--tasks plan_a,plan_b,qa]
  python run.py run --mitigate --models X   mitigation pass (multi-iteration revise loop)
  python run.py run --no-spec               same sweep with SPECULATIVE DECODING off. On by
                                             default (config.yaml's `speculative:` block) for
                                             the models that declare a `draft:`; turn it off
                                             when the run has to be token-comparable to results
                                             recorded before it existed — accepting a drafted
                                             token depends on the target's sampler agreeing,
                                             so quality numbers do not pool across the setting
  python run.py run --major cs --tasks converge --preflight   MODE C only: validate locked
                                             slots, spend no GPU time (needs --major: only
                                             schools with a *.locked_slots.yaml sibling support it)
  python run.py run --major cs --tasks converge [--variants blind,feedback] [--scenarios id]
  python run.py run --major cs --tasks plan_b_thinking   Mode B with reasoning ON, budget-
                                             capped at config.yaml's thinking.budget_tokens —
                                             own server, own results/runs_thinking.jsonl,
                                             never mixed with the --reasoning off baseline
  python run.py report                      results/report.md for the CURRENTLY CONFIGURED
                                             fixture: plan tables, Mode C, validity guards,
                                             review queue — one file, every mode
  python run.py report --all                results/summary.md aggregating every
                                             results/results_<slug>/ the multi-school run wrote
  python run.py parity                      check the vendored planner against the app's
  python run.py fixture-check               diff the plan fixture against the catalog DB

The harness starts and stops llama-server itself (see harness/server.py): under llama.cpp,
context size and GPU/KV settings are launch flags, so "every model saw identical settings"
is only true if one place owns the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _install_sigterm_handler() -> None:
    """Make an external SIGTERM unwind exactly like Ctrl-C, instead of killing the process
    outright.

    Python installs a handler for SIGINT (Ctrl-C) that raises KeyboardInterrupt, which is why
    `run.py serve`'s `except KeyboardInterrupt: handle.stop()` and `run_eval`'s
    `try/finally: handle.stop()` / `_run_lock`'s `try/finally: lock.unlink()` all work under
    Ctrl-C. SIGTERM gets no such treatment by default — the process dies immediately, the
    finally blocks never run, and the llama-server subprocess (holding several GB of VRAM) and
    the `.{tag}.lock` file are both left behind. `timeout <n> run.py ...` and a plain
    `kill <pid>` both send SIGTERM, not SIGINT, and the harness's own launch pattern (nohup +
    disown, run in the background) means there is no terminal for Ctrl-C to reach even when a
    human wants to stop it cleanly. Installing this at the very top of `main()`, before any
    subcommand runs, is what makes every one of those paths — `run`, `converge`, `serve` —
    release the GPU on the way out instead of only doing so when asked nicely.
    """
    def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def main() -> None:
    _install_sigterm_handler()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="diagnose local llama-server setup")
    checkp = sub.add_parser("check", help="validate fixture, prompts, GPU, and model files")
    checkp.add_argument(
        "--major", help="match plan_fixtures/*.yaml by slug, filename, or program name "
                        "(substring, case-insensitive) and use that school for every mode — "
                        "errors (does not prompt) if more than one fixture matches")
    reportp = sub.add_parser("report", help="generate results/report.md")
    reportp.add_argument("--all", action="store_true",
                         help="aggregate every results/results_<slug>/ into results/summary.md "
                              "instead of reporting on the single currently-configured fixture")
    sub.add_parser("parity", help="check the vendored planner against the app's")
    sub.add_parser("fixture-check", help="diff the plan fixture against the catalog DB")
    sub.add_parser("refresh-offerings",
                   help="rewrite the fixture's offered_terms from observed PurdueIO offerings")
    servep = sub.add_parser("serve", help="launch llama-server for one model and hold it")
    servep.add_argument("model", help="model name from config.yaml")
    serve_spec = servep.add_mutually_exclusive_group()
    serve_spec.add_argument("--spec", dest="spec", action="store_true", default=None,
                            help="force speculative decoding ON, overriding config.yaml")
    serve_spec.add_argument("--no-spec", dest="spec", action="store_false",
                            help="force speculative decoding OFF")

    runp = sub.add_parser(
        "run", help="every school by default (results/results_<slug>/ each); --major for one")
    runp.add_argument("--models", help="comma-separated model names (default: all)")
    runp.add_argument("--brackets", help="comma-separated brackets: 8gb,coder,server,reference")
    runp.add_argument(
        "--tasks", help="comma-separated: plan_a,plan_b,qa,explain,converge,plan_b_thinking "
                        "(default: config.yaml's run.default_tasks, which does NOT include "
                        "plan_b_thinking — opt in explicitly; converge is dropped "
                        "automatically for any school with no matching *.locked_slots.yaml)")
    runp.add_argument("--mitigate", action="store_true",
                      help="mitigation mode: multi-iteration revise loop; writes "
                           "runs_mitigated.jsonl (plan_a/plan_b/qa/explain only)")
    runp.add_argument("--major", help="run just one school instead of every school — see "
                                      "`check --major`")
    # SPECULATIVE DECODING (config.yaml's `speculative:` block; harness/server.py builds the
    # flags). On by default there for the models that declare a `draft:`; --no-spec forces it
    # off for a run that has to be comparable to results recorded before it existed, and --spec
    # forces it on for a config.yaml that has it disabled. Both reach converge and
    # plan_b_thinking too, since all three launch their servers through the same `build_argv`.
    spec_group = runp.add_mutually_exclusive_group()
    spec_group.add_argument("--spec", dest="spec", action="store_true", default=None,
                            help="force speculative decoding ON, overriding config.yaml")
    spec_group.add_argument("--no-spec", dest="spec", action="store_false",
                            help="force speculative decoding OFF — use this when the run must "
                                 "be token-comparable to a non-speculative one")
    # converge-only knobs. No-ops unless "converge" is in --tasks.
    runp.add_argument("--variants", help="converge only: comma-separated blind,feedback "
                                         "(default: config.yaml's convergence.enabled_variants)")
    runp.add_argument("--scenarios", help="converge only: comma-separated scenario ids "
                                          "(default: all)")
    runp.add_argument("--preflight", action="store_true",
                      help="converge only: validate locked slots and print the prereq-risk "
                           "note, then stop — spends no GPU time and runs no other task")

    args = ap.parse_args()

    # RESOLVED ONCE, BEFORE DISPATCH, for every subcommand that touches a fixture/database
    # (check and run both eventually call `harness.runner.load_context`, which prefers
    # MODEL_EVAL_PLAN_FIXTURE / MODEL_EVAL_REAL_DB_PROGRAM_ID over config.yaml's own paths).
    # Setting both env vars here, once, is what lets a single `--major` flag reach every
    # downstream caller without threading new parameters through `load_context`'s signature.
    #
    # `--major` selects a PLAN FIXTURE (see `resolve_major`), not a live database search any
    # more (that was `harness.real_db.resolve_major_interactive`, still there and still used by
    # `harness/real_db.py` internally — this only changed how `run.py` picks a program_id).
    # Mode A's fixture and Mode B/C's real-db program now always move together, read off the
    # SAME fixture's own `program.real_db_program_id`.
    if getattr(args, "major", None):
        fixture_path = resolve_major(args.major)
        slug, program_id, name = _fixture_meta(fixture_path)
        os.environ["MODEL_EVAL_PLAN_FIXTURE"] = str(fixture_path.relative_to(ROOT))
        if program_id:
            os.environ["MODEL_EVAL_REAL_DB_PROGRAM_ID"] = program_id
        print(f"[major] {args.major!r} -> {slug} ({name})")

    # SAME PATTERN, SAME REASON as the fixture/program env vars above: `--spec`/`--no-spec` has
    # to reach every server launch in this process — run_eval's, run_convergence's and
    # run_thinking_experiment's — and they do not share a call chain. `harness.server.
    # speculative_enabled` reads this and lets it win over config.yaml's `speculative.enabled`.
    if getattr(args, "spec", None) is not None:
        os.environ["MODEL_EVAL_SPECULATIVE"] = "1" if args.spec else "0"
        print(f"[spec] speculative decoding forced {'ON' if args.spec else 'OFF'} "
              f"for this run (config.yaml overridden)")

    if args.cmd == "doctor":
        doctor()
    elif args.cmd == "check":
        check()
    elif args.cmd == "serve":
        serve(args.model)
    elif args.cmd == "run":
        _run(args)
    elif args.cmd == "report":
        if getattr(args, "all", False):
            from harness.report import generate_summary
            generate_summary(ROOT)
        else:
            from harness.report import generate_report
            generate_report(ROOT)
    elif args.cmd == "parity":
        parity()
    elif args.cmd == "fixture-check":
        fixture_check()
    elif args.cmd == "refresh-offerings":
        refresh_offerings()


# --- plan-fixture selection (multi-school) ------------------------------------------------------
#
# `--major` USED TO mean "search the live catalog_ingestion `programs` table and point Mode B/C
# at whatever matches" (`harness.real_db.resolve_major_interactive`) while Mode A stayed on
# whatever fixture config.yaml happened to name — the two could (and did) silently describe
# different programs. As of 2026-08-08, a plan fixture is the unit of selection for every mode:
# each `plan_fixtures/*.yaml` names its own `real_db_program_id`, so picking a fixture picks
# Mode A *and* Mode B/C's program together. `harness/real_db.py` itself is untouched — it still
# does every Postgres round trip exactly as before; only WHICH program_id reaches it changed.


def discover_fixtures() -> list[Path]:
    """Every plan fixture available for `--major`/the multi-school default loop, sorted for
    stable output. `*.locked_slots.yaml` sidecars are not fixtures themselves."""
    return sorted(
        p for p in (ROOT / "plan_fixtures").glob("*.yaml")
        if not p.name.endswith(".locked_slots.yaml")
    )


def _fixture_meta(path: Path) -> tuple[str, str, str]:
    """(slug, real_db_program_id, program_name) read straight off the YAML — cheaper than a
    full `load_fixture` (no course/scenario parsing) for listing/matching."""
    import yaml
    data = yaml.safe_load(path.read_text())
    program = data.get("program") or {}
    return (str(program.get("slug") or path.stem),
            str(program.get("real_db_program_id") or ""),
            str(program.get("name") or path.stem))


def resolve_major(search_text: str) -> Path:
    """Match `--major` against the plan_fixtures list (slug, filename stem, or program name,
    case-insensitive substring) and return exactly one fixture path. No stdin prompt on an
    ambiguous match — unlike a live database's hundreds of programs, the fixture list is short
    enough to just print the candidates and ask for a more specific string, which also keeps
    this safe to call from a non-interactive/background run.
    """
    search = search_text.strip().lower()
    all_fixtures = [(p, *_fixture_meta(p)) for p in discover_fixtures()]
    # EXACT SLUG MATCH WINS OUTRIGHT, before falling back to substring matching — a short slug
    # like "me" or "ag" is otherwise a substring of unrelated program names ("Ele-ME-ntary",
    # pre-veterinary "Medi-ci-ne"), which made the intended, unique, cheap-to-type identifier
    # report a false ambiguity against schools that merely happen to contain those letters.
    exact = [(p, slug, name) for p, slug, pid, name in all_fixtures if slug.lower() == search]
    if len(exact) == 1:
        return exact[0][0]
    matches = [(p, slug, name) for p, slug, pid, name in all_fixtures
               if search in slug.lower() or search in p.stem.lower() or search in name.lower()]
    if not matches:
        listing = "\n".join(f"  - {slug} ({p.name}): {name}" for p, slug, pid, name in all_fixtures)
        raise SystemExit(f"--major {search_text!r} matched none of the available schools:\n{listing}")
    if len(matches) > 1:
        listing = "\n".join(f"  - {slug} ({p.name}): {name}" for p, slug, name in matches)
        raise SystemExit(
            f"--major {search_text!r} matched {len(matches)} schools — be more specific:\n{listing}")
    return matches[0][0]


# --- run -----------------------------------------------------------------------------------------


def _run(args) -> None:
    """One entry point for every mode, Mode C (`converge`) included — `--tasks` picks which.

    `--major` given: run that one school (see `resolve_major`), same as always. `--major`
    omitted: run EVERY school in `plan_fixtures/`, one after another, each into its own
    `results/results_<slug>/` — see `_run_all_schools`. This is a real behavior change
    (2026-08-08): the bare `run.py run` used to mean "the one fixture config.yaml currently
    names," and now means "everything" — a single school is now the thing you have to ask for.
    """
    if getattr(args, "major", None):
        _run_single(args)
    else:
        _run_all_schools(args)


def _run_single(args, *, results_dir_override: str | None = None) -> None:
    """One school, one process. `results_dir_override`, when given, points this run's output
    at `results/results_<slug>/` instead of config.yaml's own `paths.results_dir` — set by
    `_run_all_schools`, left alone (so a plain `--major` run keeps writing to the normal
    `results/`) for a single-school invocation.

    USED TO BE TWO COMMANDS (`run` and `converge`), because Mode C has its own results file,
    own lock, own per-model server-then-loop-variants shape instead of run_eval's
    per-model-then-per-task shape. That is still true underneath — `run_convergence` is a
    separate function with a separate schema, unchanged — but there is no reason a caller
    should have to know that split exists. `--tasks converge` (or the default, which includes
    it) runs it right after the other tasks, same process, same `--models`/`--brackets` filter.
    """
    cfg = _cfg()
    task_set = {t.strip() for t in args.tasks.split(",")} if args.tasks \
        else set(cfg["run"]["default_tasks"])
    if not task_set:
        raise SystemExit("No tasks left to run.")

    do_converge = "converge" in task_set
    # `plan_b_thinking` — the pre-plan-reasoning experiment (see `harness.runner.
    # run_thinking_experiment`). Same special-casing as `converge`: it needs its own server
    # launch (reasoning ON, budget-capped) and own results file, so it cannot go through
    # `run_eval`'s normal per-model loop, which launches every model with the baseline's
    # `--reasoning off`. Deliberately absent from `run.default_tasks` — see that function's
    # own docstring for why it must stay opt-in.
    do_thinking = "plan_b_thinking" in task_set
    eval_tasks = task_set - {"converge", "plan_b_thinking"}

    if args.preflight:
        # Spends no GPU time, by contract — never falls through to `run_eval` even if other
        # tasks were also requested.
        if not do_converge:
            raise SystemExit("--preflight only means something for the converge task; "
                             "add it with --tasks converge.")
        _converge_preflight()
        return

    if results_dir_override is not None:
        os.environ["MODEL_EVAL_RESULTS_DIR"] = results_dir_override

    from harness.runner import ConcurrentRunError, run_eval, run_thinking_experiment
    try:
        if eval_tasks:
            run_eval(
                ROOT,
                models=args.models.split(",") if args.models else None,
                brackets=args.brackets.split(",") if args.brackets else None,
                tasks=list(eval_tasks),
                mitigate=args.mitigate,
            )
        if do_converge:
            _converge_run(args)
        if do_thinking:
            run_thinking_experiment(
                ROOT,
                models=args.models.split(",") if args.models else None,
                brackets=args.brackets.split(",") if args.brackets else None,
            )
    except ConcurrentRunError as exc:
        # A guard, not a crash — print the reason and the fix, not a traceback.
        print(f"refusing to start: {exc}")
        sys.exit(1)


def _run_all_schools(args) -> None:
    """`run.py run` with no `--major`: every plan fixture, one after another, each writing to
    its own `results/results_<slug>/`.

    Mode C is dropped from a school's OWN task set (not the whole sweep's) when that fixture has
    no `<stem>.locked_slots.yaml` sibling — most schools don't have one (see
    `harness/convergence.locked_slots_path_for`), and erroring the entire sweep over a task only
    a couple of sixteen fixtures currently support would be worse than skipping it there and
    saying so.

    One school's failure (bad fixture, a crash mid-run, a concurrent-run lock) is caught and
    reported, not raised — a broken school should not cost every other school its results. This
    is the same shape a hand-written shell loop over `run.py run --major <x>` would have; the
    difference is it is now what the bare command does.
    """
    import copy

    fixtures = discover_fixtures()
    if not fixtures:
        raise SystemExit("plan_fixtures/ has no fixtures to run.")

    requested_tasks = {t.strip() for t in args.tasks.split(",")} if args.tasks else None
    default_tasks = set(_cfg()["run"]["default_tasks"])
    failed: list[str] = []

    for path in fixtures:
        slug, program_id, name = _fixture_meta(path)
        print(f"\n{'=' * 70}\nSCHOOL: {slug} — {name} ({path.name})\n{'=' * 70}")

        os.environ["MODEL_EVAL_PLAN_FIXTURE"] = str(path.relative_to(ROOT))
        if program_id:
            os.environ["MODEL_EVAL_REAL_DB_PROGRAM_ID"] = program_id
        else:
            print(f"  [WARN] {slug}: no real_db_program_id set on this fixture — Mode B/C "
                  f"will read whatever program config.yaml's real_db.program_id last pointed "
                  f"at, almost certainly the wrong one for this school.")

        task_set = set(requested_tasks) if requested_tasks is not None else set(default_tasks)
        if "converge" in task_set:
            from harness.convergence import locked_slots_path_for
            from harness.fixtures import load_fixture
            if locked_slots_path_for(load_fixture(path)) is None:
                print(f"  [skip] {slug}: no locked-slots file — dropping `converge` for this "
                      f"school only.")
                task_set.discard("converge")

        school_args = copy.copy(args)
        school_args.tasks = ",".join(sorted(task_set)) if task_set else ""
        try:
            _run_single(school_args, results_dir_override=f"results/results_{slug}")
        except SystemExit as exc:
            print(f"  [FAILED] {slug}: {exc}")
            failed.append(slug)
        except Exception as exc:  # one bad school must not sink the sweep
            print(f"  [FAILED] {slug}: {type(exc).__name__}: {exc}")
            failed.append(slug)

    print(f"\n{'=' * 70}")
    print(f"ALL SCHOOLS DONE — failures: {', '.join(failed)}" if failed
          else "ALL SCHOOLS DONE — no failures")
    print(f"{'=' * 70}")


def _cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# --- converge (MODE C) --------------------------------------------------------------------------


def _converge_preflight() -> None:
    """``run --tasks converge --preflight``: spends no GPU time. Run this after editing locks."""
    from harness.convergence import load_locked_slots, locked_slots_path_for, preflight, prereq_risk_note
    from harness.runner import load_context

    ctx = load_context(ROOT)
    conv_cfg = ctx.cfg.get("convergence") or {}
    if not conv_cfg:
        print("config.yaml has no `convergence:` block.")
        sys.exit(1)
    slots_path = locked_slots_path_for(ctx.fixture)
    if slots_path is None:
        print(f"no locked-slots file for {ctx.fixture.path.name} — Mode C isn't supported for "
              f"this fixture (needs {ctx.fixture.path.with_suffix('').name}.locked_slots.yaml).")
        sys.exit(1)
    locked = load_locked_slots(slots_path)
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


def _converge_run(args) -> None:
    """Mode C proper. Raises `ConcurrentRunError` up to `_run`'s shared try/except — same
    handling as every other task, since it is now one more thing `run` can be asked to do."""
    from harness.convergence import run_convergence

    split = lambda v: v.split(",") if v else None
    run_convergence(
        ROOT,
        models=split(args.models),
        brackets=split(args.brackets),
        variants=split(args.variants),
        scenarios=split(args.scenarios),
    )


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
    from harness.catalog_export import integrity_problems
    from harness.fixtures import load_fixture
    from harness.planner import generate_plan
    from harness.plan_scorers import score_plan, plan_to_semesters
    from harness.prompts import PromptBuilder
    from harness.real_db import (
        REQUIREMENT_SCOPE_ALIASES, build_scenario_database, load_real_db,
        real_db_base_from_fixture,
    )
    from harness.server import gpu_memory_used_mb, slot_context

    cfg = _cfg()
    fixture_rel = os.environ.get("MODEL_EVAL_PLAN_FIXTURE", cfg["paths"]["plan_fixture"])
    fixture = load_fixture(ROOT / fixture_rel)
    # Per-SLOT, matching `runner.load_context` exactly — see its comment for why deriving this
    # from num_ctx silently over-fills Mode B's prompt the moment `parallel` goes above 1.
    budget_tokens = slot_context(cfg["run"]) - cfg["run"]["max_plan_tokens"]
    if fixture.source == "fixture":
        # No Postgres for a `program.source: fixture` pseudo-major — see
        # `harness.runner.load_context`'s identical branch and
        # `real_db.real_db_base_from_fixture`'s docstring.
        program_id = f"fixture:{fixture.slug}"
        database = build_scenario_database(
            real_db_base_from_fixture(fixture), budget_tokens=budget_tokens
        )
    else:
        # MODEL_EVAL_REAL_DB_URL / _PROGRAM_ID override config.yaml's real_db.* — set by
        # `--major` (see main()) or by hand for a one-off check against a specific program.
        real_db_cfg = cfg["real_db"]
        pg_url = os.environ.get("MODEL_EVAL_REAL_DB_URL", real_db_cfg["url"])
        program_id = os.environ.get("MODEL_EVAL_REAL_DB_PROGRAM_ID", real_db_cfg["program_id"])
        database = load_real_db(
            pg_url, program_id,
            broaden_subjects=(tuple(real_db_cfg["broaden_subjects"])
                             if real_db_cfg.get("broaden_subjects") else None),
            force_selective_groups=tuple(real_db_cfg.get("force_selective_groups") or ()),
            manual_course_aliases=real_db_cfg.get("manual_course_aliases") or {},
            budget_tokens=budget_tokens,
            scope_filter=REQUIREMENT_SCOPE_ALIASES.get(fixture.requirement_scope),
        )
    prompts = PromptBuilder(fixture, database)

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

    # THE REAL DATABASE — what Mode B is actually shown, read live from Postgres (see
    # harness/real_db.py). Checked for referential integrity the same way the old JSON mock
    # was — a dangling reference here would mean a real_db.py bug, not bad source data.
    #
    # SCORED AGAINST THIS, NOT THE FIXTURE (see harness/real_scoring.py) — so unlike before
    # 2026-08-04, a `real_db.program_id`/`--major` that names a different program than the
    # fixture no longer corrupts Mode B's coverage/hallucination numbers; they honestly
    # describe whatever program this is. What CAN still be wrong: the fixture's own
    # scenarios (student names, completed courses, credit targets — Mode A/C context) were
    # written for one specific program, so a mismatch here means Mode B is being tested
    # against a program those scenarios don't describe, not that its numbers are meaningless.
    print(f"\ndatabase: program {program_id} (hash {database.db_hash})")
    print("  " + " | ".join(f"{t} {len(database.rows(t))}" for t in database.tables))
    problems += [f"database: {p}" for p in integrity_problems(database)]
    program_name = next((r.get("name") for r in database.rows("programs")), None)
    if program_name and program_name != fixture.name:
        print(f"  ⚠️  database program ({program_name!r}) is not the fixture's own program "
              f"({fixture.name!r}) — Mode B's numbers are honest about {program_name!r}, but "
              f"the scenarios were written for {fixture.name!r}.")

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
        approx_tokens, check_slot_context,
    )
    # PER-SLOT, for the same reason `budget_tokens` above is: at `parallel: 4` the server's
    # total context is four students' worth, and a prompt is only ever shown to one of them.
    slot = slot_context(cfg["run"])
    budget = slot - cfg["run"]["max_plan_tokens"]
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

    problems += _check_speculative(cfg, root)

    gpu = gpu_memory_used_mb()
    print(f"\nnvidia-smi: {'available, ' + str(gpu) + ' MB used' if gpu else 'NOT available'}")

    if problems:
        print(f"\n❌ {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\n✅ check passed. Run `python run.py doctor` before the first run on a new box.")


def _check_speculative(cfg: dict, models_root: Path) -> list[str]:
    """Is every model's draft pairing one llama-server will actually accept?

    THE FAILURE THIS EXISTS TO MOVE EARLIER. A draft whose vocabulary doesn't match its target's
    is not a warning in llama.cpp — the server aborts, and it aborts at load time, several
    minutes into the run, once per model, where it reads as "that model is broken" rather than
    "that draft doesn't belong to it". Vocabulary sizes come from the ggufs' own headers
    (`gguf_vocab_size`), so this checks the files on disk rather than trusting config.yaml's
    comment about which families share a tokenizer.

    The 100-token tolerance is llama.cpp's own (`SPEC_VOCAB_MAX_SIZE_DIFFERENCE`): a draft may
    differ by a handful of added special tokens, but not by a whole tokenizer.
    """
    from harness.server import gguf_vocab_size, resolve_draft, speculative_enabled

    problems: list[str] = []
    if not speculative_enabled(cfg):
        print("\nspeculative decoding: OFF "
              "(config.yaml `speculative.enabled: false`, or MODEL_EVAL_SPECULATIVE=0)")
        return problems

    spec = cfg.get("speculative") or {}
    print(f"\nspeculative decoding: ON — draft {spec.get('draft_gguf')!r} "
          f"(n_max {spec.get('n_max', 4)}, p_min {spec.get('p_min', 0.75)})")
    vocab_cache: dict[Path, int | None] = {}

    def vocab(path: Path) -> int | None:
        if path not in vocab_cache:
            vocab_cache[path] = gguf_vocab_size(path)
        return vocab_cache[path]

    for model in cfg["models"]:
        try:
            draft_path = resolve_draft(cfg, model)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if draft_path is None:
            print(f"  {model['name']:22s} no draft — plain decoding")
            continue
        if not draft_path.exists():
            problems.append(
                f"model {model['name']} speculates against {draft_path}, which does not exist. "
                f"Download it, or drop `draft:` from that entry."
            )
            print(f"  {model['name']:22s} MISSING DRAFT {draft_path}")
            continue
        target_vocab, draft_vocab = vocab(models_root / model["gguf"]), vocab(draft_path)
        if target_vocab is None or draft_vocab is None:
            # Unreadable header. Not a hard failure — the ggufs may still be fine and only the
            # cheap metadata scan gave up — but say so rather than implying it was verified.
            print(f"  {model['name']:22s} draft {draft_path.name} "
                  f"(vocab UNVERIFIED — could not read a gguf header)")
            continue
        delta = abs(target_vocab - draft_vocab)
        if delta > 100:
            problems.append(
                f"model {model['name']} (vocab {target_vocab}) cannot be drafted for by "
                f"{draft_path.name} (vocab {draft_vocab}) — llama.cpp allows a difference of "
                f"at most 100 tokens and aborts the server at load otherwise. Use a draft from "
                f"the same model family, or drop `draft:` from that entry."
            )
            print(f"  {model['name']:22s} VOCAB MISMATCH {target_vocab} vs {draft_vocab}")
        else:
            print(f"  {model['name']:22s} draft {draft_path.name} "
                  f"(vocab {target_vocab}{'' if delta == 0 else f' vs {draft_vocab}'} ✓)")
    return problems


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
