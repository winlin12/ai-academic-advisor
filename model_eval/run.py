#!/usr/bin/env python3
"""Evaluation harness CLI. Standalone: Python 3.10+, PyYAML and psycopg, nothing from the app.

  python run.py doctor                      diagnose local vLLM reachability
  python run.py check                       validate the program's data, prompts, GPU, models
  python run.py check --major "nursing"        same, against any program in the catalog —
                                             matches `programs.name` or its slug (substring,
                                             case-insensitive; also works on `run`). EVERY
                                             crawled program is selectable: courses,
                                             requirement groups and students all come from the
                                             database now, so nothing has to be hand-authored
                                             first (see harness/fixtures.py)
  python run.py serve <model>               launch vLLM for one model and hold it
  python run.py run                         THE SWEEP: two curated majors from every school
                                             (config.yaml `sweep.curated`) plus two random
                                             programs never run here before, each into its own
                                             results/results_<slug>/ — see run.sweep_programs
  python run.py run --major cs              just one program (see `check --major` above)
  python run.py run [--models a,b] [--brackets 8gb,coder] [--tasks plan_a,plan_b,qa]
  python run.py run --mitigate --models X   mitigation pass (multi-iteration revise loop)
  python run.py run --major cs --tasks converge --preflight   MODE C only: check every
                                             scenario is solvable, spend no GPU time
  python run.py run --major cs --tasks converge [--variants blind,feedback] [--scenarios id]
  python run.py report                      results/report.md for the CURRENTLY CONFIGURED
                                             program: plan tables, Mode C, validity guards,
                                             review queue — one file, every mode
  python run.py report --all                results/summary.md aggregating every
                                             results/results_<slug>/ the sweep wrote — every
                                             arm in one file
  python run.py parity                      check the vendored planner against the app's

The harness starts and stops the vLLM server itself (see harness/server.py): context length,
KV dtype, the memory fraction and the concurrent-sequence bound are all launch flags, so
"every model saw identical settings" is only true if one place owns the command line.

BACKEND SWITCHED TO vLLM 2026-08-22, from llama.cpp. Nothing recorded before that date pools
with anything recorded after it — different engine, different weights (W4A16
compressed-tensors, not GGUF), different timing source. See config.yaml's `vllm:` block.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    finally blocks never run, and the vLLM subprocess (holding several GB of VRAM) and
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

    doctorp = sub.add_parser("doctor", help="diagnose the local inference server setup")
    checkp = sub.add_parser("check", help="validate the program data, prompts, GPU, models")
    checkp.add_argument(
        "--major", help="match any crawled program by name or slug (substring, case-"
                        "insensitive) and use it for every mode — errors (does not prompt) "
                        "if more than one program matches")
    reportp = sub.add_parser("report", help="generate results/report.md")
    reportp.add_argument("--all", action="store_true",
                         help="aggregate every results/results_<slug>/ into results/summary.md "
                              "instead of reporting on the single currently-configured fixture")
    sub.add_parser("parity", help="check the vendored planner against the app's")
    servep = sub.add_parser("serve", help="launch the vLLM server for one model and hold it")
    servep.add_argument("model", help="model name from config.yaml")

    runp = sub.add_parser(
        "run", help="the curated + random sweep by default (results/results_<slug>/ each); "
                    "--major for one program")
    runp.add_argument("--models", help="comma-separated model names (default: all)")
    runp.add_argument("--brackets", help="comma-separated brackets: 8gb,coder,server,reference")
    runp.add_argument(
        "--tasks", help="comma-separated: plan_a,plan_b,qa,explain,converge "
                        "(default: config.yaml's run.default_tasks)")
    runp.add_argument("--mitigate", action="store_true",
                      help="mitigation mode: multi-iteration revise loop; writes "
                           "runs_mitigated.jsonl (plan_a/plan_b/qa/explain only)")
    runp.add_argument("--major", help="run one program instead of the sweep — see "
                                      "`check --major`")
    # SPECULATIVE DECODING, per run. config.yaml carries each model's own setting; this
    # overrides it for one invocation so the same sweep can be re-run under a different
    # proposer without editing the file. `none` forces plain decoding, which is what you want
    # for a run that has to be token-comparable to one recorded before speculation existed —
    # accepting a drafted token depends on the target's sampler agreeing, so quality numbers
    # do not pool across the setting.
    runp.add_argument("--spec", choices=["none", "mtp", "ngram", "draft"], default=None,
                      help="override config.yaml's speculative method for this run")

    # converge-only knobs. No-ops unless "converge" is in --tasks.
    runp.add_argument("--variants", help="converge only: comma-separated blind,feedback "
                                         "(default: config.yaml's convergence.enabled_variants)")
    runp.add_argument("--scenarios", help="converge only: comma-separated scenario ids "
                                          "(default: all)")
    runp.add_argument("--preflight", action="store_true",
                      help="converge only: validate locked slots and print the prereq-risk "
                           "note, then stop — spends no GPU time and runs no other task")

    # WHICH ENGINE, for every subcommand that launches or inspects a server. Added to the
    # PARSER ROOT rather than to `run` alone so `doctor`, `check` and `serve` all answer about
    # the engine you are actually going to use — a `doctor` that always reported on vLLM while
    # `run` launched llama.cpp would be worse than no doctor.
    for p in (runp, checkp, servep, doctorp):
        p.add_argument(
            "--engine", choices=["llamacpp", "vllm"], default=None,
            help="inference server to use (default: config.yaml's `engine:`, currently "
                 "llamacpp). llama.cpp loads the model's `gguf:`; vLLM loads its `model_path:`")

    args = ap.parse_args()

    # RESOLVED ONCE, BEFORE DISPATCH, for every subcommand that touches the database (`check`
    # and `run` both end up in `harness.runner.load_context`, which prefers
    # MODEL_EVAL_REAL_DB_PROGRAM_ID over config.yaml's own `real_db.program_id`). Setting the
    # env var here, once, is what lets a single `--major` flag reach every downstream caller
    # without threading a new parameter through `load_context`'s signature.
    #
    # `--major` IS A LIVE DATABASE SEARCH AGAIN (2026-08-12), over every plannable program in
    # the catalog — see `resolve_major`. It briefly meant "pick one of the nineteen
    # `plan_fixtures/*.yaml` files", which is exactly the limit that got removed: there is no
    # hand-authored file to write for a program any more, so any of the ~950 crawled programs
    # is a valid `--major`.
    if getattr(args, "major", None):
        program = resolve_major(args.major)
        os.environ["MODEL_EVAL_REAL_DB_PROGRAM_ID"] = program["id"]
        print(f"[major] {args.major!r} -> {program['name']} "
              f"[{program['slug']}] ({program['n_options']} requirement options)")

    # SAME PATTERN, SAME REASON as `--major` above: `--spec` has to reach every server launch
    # in this process, and they do not share a call
    # chain. `harness.server._speculative_config` reads this and lets it win over config.yaml.
    # ALWAYS EXPORTED, resolved or defaulted — not only when `--engine` is passed. Several
    # readers (notably `harness.server.slot_context`, which every caller hands `cfg["run"]` and
    # not the whole config) can only see this through the environment, so leaving it unset when
    # the flag is absent would make them silently assume a default that config.yaml may have
    # overridden.
    _cfg_engine = (_cfg().get("engine") or "llamacpp")
    os.environ["MODEL_EVAL_ENGINE"] = getattr(args, "engine", None) or _cfg_engine
    if getattr(args, "engine", None):
        print(f"[engine] {args.engine!r} for this run (config.yaml's {_cfg_engine!r} overridden)")

    if getattr(args, "spec", None) is not None:
        os.environ["MODEL_EVAL_SPEC"] = args.spec
        print(f"[spec] speculative decoding forced to {args.spec!r} for this run "
              f"(config.yaml overridden)")

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


# --- program selection --------------------------------------------------------------------------
#
# EVERY PROGRAM IN THE CATALOG IS SELECTABLE. `--major` searches the live `programs` table
# (`harness.real_db.resolve_program`) and the program it lands on is the whole selection: Mode A,
# Mode B and Mode C all read that one program's own database, and the students are synthesized
# from it (`fixtures.synthesize_scenarios`).
#
# It used to search `plan_fixtures/*.yaml` instead, because Mode A needed a hand-authored course
# catalog and hand-written scenarios that only existed for nineteen programs. Both now come from
# the crawl, so the fixture files — and the ceiling they imposed on which majors could be
# evaluated at all — are gone.


def _llamacpp_env(cfg: dict) -> dict:
    """`llama-server`'s environment for a one-shot probe — the CUDA libs from the pip-wheel
    toolkit, which are not on the system linker path. Mirrors `harness.server._server_env`."""
    from harness.server import _server_env  # noqa: PLC0415
    return _server_env(cfg)


def _pg_url() -> str:
    return os.environ.get("MODEL_EVAL_REAL_DB_URL", _cfg()["real_db"]["url"])


def resolve_major(search_text: str) -> dict:
    """`--major <text>` -> one program row (id/name/slug/n_options), or exit with the
    candidates listed. Non-interactive by design: a sweep runs unattended, so an ambiguous
    match has to fail loudly rather than block on stdin."""
    from harness.real_db import program_slug, resolve_program

    try:
        row = resolve_program(_pg_url(), search_text)
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    return {**row, "slug": program_slug(row["name"])}


def sweep_programs() -> list[dict]:
    """The programs `run.py run` (no `--major`) evaluates: config.yaml's curated picks, plus
    `sweep.random_untested` programs drawn at random from everything else in the catalog.

    TWO CURATED MAJORS PER SCHOOL, because a sweep has to stay comparable run over run — those
    are the rows whose history means something, and they are spread across schools so no one
    college's quirks stand in for the university. They are named, not derived: the crawl carries
    no college column (every `programs.college_id` is NULL), so which school a program belongs
    to is knowledge that lives in `config.yaml`'s `sweep.curated` map and nowhere else.

    PLUS TWO AT RANDOM, drawn fresh each sweep from programs that have never been run here —
    no `results/results_<slug>/` directory, and not curated. This is the half that finds what
    curation cannot: the curated sixteen were all chosen when somebody was willing to hand-write
    a fixture for them, which selects for tidy, well-crawled programs. A random draw is the only
    thing that puts the harness in front of a program nobody groomed.

    Random picks are logged with their seed so a surprising result can be reproduced exactly:
    `--major <slug>` re-runs any one of them.
    """
    import random

    from harness.real_db import list_programs, program_slug, resolve_program

    cfg = _cfg()
    sweep_cfg = cfg.get("sweep") or {}
    pg_url = _pg_url()
    picks: list[dict] = []
    seen_ids: set[str] = set()

    for school, names in sorted((sweep_cfg.get("curated") or {}).items()):
        for name in names:
            try:
                row = resolve_program(pg_url, name)
            except LookupError as exc:
                print(f"  [WARN] curated {school}/{name!r}: {exc.args[0].splitlines()[0]}")
                continue
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            picks.append({**row, "slug": program_slug(row["name"]), "school": school,
                          "why": "curated"})

    wanted_random = int(sweep_cfg.get("random_untested", 2))
    if wanted_random > 0:
        results_root = ROOT / "results"
        already = {d.name[len("results_"):] for d in results_root.glob("results_*")
                   if d.is_dir()}
        pool = [
            {**row, "slug": program_slug(row["name"])}
            for row in list_programs(pg_url)
            if row["id"] not in seen_ids and program_slug(row["name"]) not in already
        ]
        seed = int.from_bytes(os.urandom(4), "big")
        rng = random.Random(seed)
        drawn = rng.sample(pool, min(wanted_random, len(pool)))
        print(f"[sweep] {len(pool)} untested programs in the catalog; drawing "
              f"{len(drawn)} at random (seed {seed})")
        for row in drawn:
            picks.append({**row, "school": "(random)", "why": "random, never run here"})

    return picks


# --- run -----------------------------------------------------------------------------------------


def _run(args) -> None:
    """One entry point for every mode, Mode C (`converge`) included — `--tasks` picks which.

    `--major` given: run that one program (see `resolve_major`). `--major` omitted: run the
    sweep — two curated majors from every school plus two random untested programs, each into
    its own `results/results_<slug>/` — see `sweep_programs`/`_run_sweep`.
    """
    if getattr(args, "major", None):
        _run_single(args)
    else:
        _run_sweep(args)


def _run_single(args, *, results_dir_override: str | None = None) -> None:
    """One school, one process. `results_dir_override`, when given, points this run's output
    at `results/results_<slug>/` instead of config.yaml's own `paths.results_dir` — set by
    `_run_sweep`, left alone (so a plain `--major` run keeps writing to the normal
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
    # `plan_b_thinking` REMOVED 2026-08-24 — the pre-plan-reasoning experiment. See
    # config.yaml's `thinking:` note for the measurements that retired it. A stale `--tasks
    # plan_b_thinking` is now simply an unknown task and falls out of `task_set` below.
    eval_tasks = task_set - {"converge"}

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

    from harness.runner import ConcurrentRunError, run_eval
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
    except ConcurrentRunError as exc:
        # A guard, not a crash — print the reason and the fix, not a traceback.
        print(f"refusing to start: {exc}")
        sys.exit(1)


def _run_sweep(args) -> None:
    """`run.py run` with no `--major`: the curated two-per-school set plus a couple of random
    untested programs (see `sweep_programs`), one after another, each writing to its own
    `results/results_<slug>/`.

    Every task runs for every program. Mode C used to be dropped for programs with no
    hand-authored locked-slots file, which was most of them; student-authored locks are gone
    (see `harness.convergence`'s `LockedSlot` note) and Mode C now starts from an empty set,
    so there is nothing left to opt into per program.

    One program's failure (a crash mid-run, a concurrent-run lock, a program whose crawl is too
    thin to plan) is caught and reported, not raised — a broken program must not cost every
    other one its results. That matters more now that two of the picks are random: an
    ungroomed program failing is a FINDING, and the sweep has to survive producing it.
    """
    import copy

    picks = sweep_programs()
    if not picks:
        raise SystemExit("no programs to run — check config.yaml's `sweep.curated` and that "
                         "the catalog database has been crawled.")

    requested_tasks = {t.strip() for t in args.tasks.split(",")} if args.tasks else None
    default_tasks = set(_cfg()["run"]["default_tasks"])
    failed: list[str] = []

    print(f"\n[sweep] {len(picks)} program(s):")
    for pick in picks:
        print(f"  - {pick['school']:14s} {pick['name']}  [{pick['slug']}] ({pick['why']})")

    for pick in picks:
        print(f"\n{'=' * 70}\nPROGRAM: {pick['slug']} — {pick['name']} "
              f"({pick['school']}, {pick['why']})\n{'=' * 70}")
        os.environ["MODEL_EVAL_REAL_DB_PROGRAM_ID"] = pick["id"]

        task_set = set(requested_tasks) if requested_tasks is not None else set(default_tasks)
        school_args = copy.copy(args)
        school_args.tasks = ",".join(sorted(task_set)) if task_set else ""
        try:
            # RESULTS ROOT IS OVERRIDABLE so two arms of one experiment can be kept apart.
            # `_run_single` sets MODEL_EVAL_RESULTS_DIR per program, which silently CLOBBERS an
            # outer MODEL_EVAL_RESULTS_DIR — so setting that env var around a full sweep did
            # nothing and both arms wrote to the same `results/results_<slug>/` paths.
            # MEASURED 2026-08-25: an advising-thinking A/B appended both arms into one file
            # and the comparison had to be reconstructed from `reasoning_chars`.
            # `MODEL_EVAL_RESULTS_ROOT` is the knob that actually works; it defaults to
            # `results`, so a plain `run.py run` is unchanged.
            root_dir = os.environ.get("MODEL_EVAL_RESULTS_ROOT", "results")
            _run_single(school_args,
                        results_dir_override=f"{root_dir}/results_{pick['slug']}")
        except SystemExit as exc:
            print(f"  [FAILED] {pick['slug']}: {exc}")
            failed.append(pick["slug"])
        except Exception as exc:  # one bad program must not sink the sweep
            print(f"  [FAILED] {pick['slug']}: {type(exc).__name__}: {exc}")
            failed.append(pick["slug"])

    print(f"\n{'=' * 70}")
    print(f"SWEEP DONE — failures: {', '.join(failed)}" if failed
          else "SWEEP DONE — no failures")
    print(f"{'=' * 70}")


def _cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# --- converge (MODE C) --------------------------------------------------------------------------


def _converge_preflight() -> None:
    """``run --tasks converge --preflight``: spends no GPU time.

    All that is left to check is REACHABILITY — student-authored locked slots are gone (see
    `harness.convergence`'s `LockedSlot` note), so there are no pins to validate and Mode C
    runs for every program starting from an empty set.
    """
    from harness.convergence import preflight, prereq_risk_note
    from harness.runner import load_context

    ctx = load_context(ROOT)
    conv_cfg = ctx.cfg.get("convergence") or {}
    if not conv_cfg:
        print("config.yaml has no `convergence:` block.")
        sys.exit(1)
    print(prereq_risk_note(ctx.fixture, conv_cfg))
    problems = preflight(ctx, conv_cfg)
    if problems:
        print(f"\n❌ {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\n✅ preflight passed: every scenario is solvable by the deterministic planner.")


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


def _check_draft_models(cfg: dict, models_root: Path) -> list[str]:
    """Is every `draft_model:` pairing one that can actually speculate for its target?

    THE FAILURE THIS EXISTS TO MOVE EARLIER, and it is worse here than it was under llama.cpp.
    There, a draft whose vocabulary did not match its target aborted the server at load — loud,
    immediate, unmissable. vLLM does not abort: a draft with a different tokenization proposes
    token IDS that mean something else to the target, every proposal is rejected, and the run
    completes normally having spent extra compute to produce exactly the baseline's output. The
    symptom is "speculative decoding did nothing", which reads as a property of the workload
    rather than as a broken pairing.

    CHECKED BY ENCODING, NOT BY COUNTING. Two models can report the same `vocab_size` and still
    tokenize differently — same size is necessary, not sufficient — so this runs a real string
    through both tokenizers and compares the ids.

    Only models that declare a `draft_model:` are checked; the rest simply cannot use
    `--spec draft` and `server._speculative_config` says so when asked.
    """
    problems: list[str] = []
    pairs = [(m, m.get("draft_model")) for m in cfg["models"] if m.get("draft_model")]
    if not pairs:
        print("\ndraft models: none configured (`--spec draft` unavailable; ngram needs none)")
        return problems

    print("\ndraft models:")
    probe = ("Place CS 25100 in semester 3; prerequisites CS 18200 and CS 24000 "
             "must come earlier.")
    for model, draft in pairs:
        path = models_root / draft
        if not path.is_dir():
            problems.append(
                f"model {model['name']} declares draft_model {draft}, which is not on disk at "
                f"{path}. Download it, or drop `draft_model:` from that entry."
            )
            print(f"  {model['name']:16s} MISSING {path}")
            continue
        # THE TOKENIZER LIVES IN THE HF CHECKPOINT, NOT THE GGUF. `model_path` is the .gguf
        # under the default engine and `AutoTokenizer` cannot read one, so this deliberately
        # reads `model_path_vllm` regardless of which engine is active — the question it asks
        # ("do target and draft tokenize identically") is a property of the weights, not of the
        # server. Speculative decoding is a vLLM feature anyway; llama.cpp entries simply have
        # nothing to verify here.
        tok_dir = model.get("model_path_vllm")
        if not tok_dir:
            print(f"  {model['name']:16s} draft {draft} (UNVERIFIED — no model_path_vllm, "
                  f"so no tokenizer to compare)")
            continue
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415 - heavy, only needed here
            a = AutoTokenizer.from_pretrained(str(models_root / tok_dir))
            b = AutoTokenizer.from_pretrained(str(path))
            same = a(probe)["input_ids"] == b(probe)["input_ids"]
        except Exception as exc:  # noqa: BLE001 - an unreadable tokenizer is not a hard failure
            print(f"  {model['name']:16s} draft {draft} (UNVERIFIED — {type(exc).__name__})")
            continue
        if not same:
            problems.append(
                f"model {model['name']} cannot be drafted for by {draft}: the two tokenize the "
                f"same string differently, so every proposal would be rejected and the run "
                f"would silently measure the baseline at extra cost."
            )
            print(f"  {model['name']:16s} TOKENIZER MISMATCH vs {draft}")
        else:
            gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
            print(f"  {model['name']:16s} draft {draft} ({gb:.1f} GB, tokenizer ✓)")
    return problems


def _check_transformers(version: str) -> None:
    """Warn about a transformers new enough to break `gemma4-26b`.

    CHECKED HERE BECAUSE THE REAL FAILURE IS UNREADABLE. vLLM 0.27.1 requires only
    `transformers>=5.5.3`, so a fresh `pip install vllm` pulls whatever is newest — and from
    5.15.0 onward that refuses to answer `config.head_dim` globally for architectures that
    declare it per layer. Gemma 4 does (`global_head_dim: 512` on its full-attention layers,
    `head_dim: 256` on the sliding ones), vLLM's `get_head_size()` asks globally anyway, and
    the result is `AmbiguousGlobalPerLayerAttributeError` several hundred lines into a
    traceback — which through `model_manager` (stderr to DEVNULL) is not visible at all and
    surfaces only as a model switch that times out after ten minutes.

    BISECTED 2026-08-22: 5.14.1 works, 5.15.0 is the first broken release, qwen3.8-27b is
    unaffected on either. A WARNING rather than a hard failure — a future release may well fix
    it from the vLLM side, and this should not block a run that would have worked.
    """
    try:
        parts = tuple(int(x) for x in version.split(".")[:2])
    except ValueError:
        return
    if parts >= (5, 15):
        print(f"  ⚠ transformers {version} is >= 5.15.0, which breaks gemma4-26b at config "
              f"parse (AmbiguousGlobalPerLayerAttributeError on 'head_dim'). "
              f"Pin it: pip install 'transformers==5.14.1'")


def doctor() -> None:
    """The most likely reasons a run fails on this box, diagnosed explicitly.

    vLLM runs as a native Linux process here, driven by `vllm serve` out of the project venv,
    so there is no cross-machine networking to diagnose — just: is the launcher there, can it
    import vLLM against a CUDA-visible GPU, and is the configured port free (or already
    serving what you expect).
    """
    from harness.server import backend_cfg, engine, gpu_memory_used_mb, resolve_host

    cfg = _cfg()
    eng = engine(cfg)
    backend = backend_cfg(cfg)
    port = backend["port"]
    host = resolve_host(backend.get("host", "auto"))
    print(f"=== local {eng} setup ===\n")

    exe = Path(backend["server_exe"])
    print(f"{eng} launcher: {'found' if exe.exists() else 'NOT FOUND at ' + str(exe)}")
    if not exe.exists() and eng == "llamacpp":
        print("  Build it: source .llamacpp-env.sh && cmake -B build -DGGML_CUDA=ON \\")
        print("              -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j 6")
    elif not exe.exists():
        print("  Install it into the project venv: .venv/bin/pip install vllm==0.27.1")
    elif eng == "llamacpp":
        # THE REAL CHECK FOR llama.cpp IS THAT IT CAN FIND ITS CUDA LIBRARIES, since they come
        # from pip wheels rather than a system install — `--version` fails outright if not.
        import subprocess
        probe = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                               timeout=60, env=_llamacpp_env(cfg))
        out = (probe.stdout + probe.stderr).strip().splitlines()
        print(f"  {out[0] if out else 'no output'}")
        dev = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True,
                             timeout=60, env=_llamacpp_env(cfg))
        for line in (dev.stdout + dev.stderr).splitlines():
            if "CUDA" in line or "no devices" in line.lower():
                print(f"  {line.strip()}")
    else:
        # IMPORTING vLLM IS THE REAL CHECK, not the launcher's existence. The wheel is
        # torch-plus-CUDA-libraries deep, and the way this fails on a box whose driver, CUDA
        # runtime and torch build have drifted apart is an ImportError several seconds into a
        # `vllm serve` that then looks like a model problem. Better to find it here.
        import subprocess
        probe = subprocess.run(
            [str(exe.parent / "python"), "-c",
             "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, "
             "torch.cuda.is_available(), torch.cuda.get_device_name(0) "
             "if torch.cuda.is_available() else 'no-cuda', transformers.__version__)"],
            capture_output=True, text=True, timeout=180)
        if probe.returncode == 0:
            print(f"  imports OK: {probe.stdout.strip()}")
            _check_transformers(probe.stdout.split()[-1])
        else:
            print(f"  IMPORT FAILED: {probe.stderr.strip().splitlines()[-1:]}")

    models_root = Path(backend["models_root"])
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
    from harness.planner import generate_plan
    from harness.plan_scorers import score_plan, plan_to_semesters
    from harness.prompts import PromptBuilder
    from harness.real_db import fixture_from_database, load_real_db
    from harness.server import gpu_memory_used_mb, gpu_total_mb, slot_context

    cfg = _cfg()
    # Per-SLOT, matching `runner.load_context` exactly — see its comment for why deriving this
    # from num_ctx silently over-fills Mode B's prompt the moment `parallel` goes above 1.
    budget_tokens = slot_context(cfg["run"]) - cfg["run"]["max_plan_tokens"]
    # MODEL_EVAL_REAL_DB_URL / _PROGRAM_ID override config.yaml's real_db.* — set by `--major`
    # (see main()) or by hand for a one-off check against a specific program.
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
    )
    # THE SAME OBJECT `runner.load_context` builds, from the same database — courses,
    # requirement groups and the five synthesized students. There is no fixture file to
    # validate any more; what this section checks is that the CRAWL is coherent enough to
    # evaluate against (see `fixtures.py`'s module docstring).
    fixture = fixture_from_database(database)
    prompts = PromptBuilder(fixture, database)

    print(f"program: {fixture.name} [{fixture.slug}]")
    edges = sum(len(c.prereqs) for c in fixture.catalog)
    print(f"  db hash {fixture.fixture_hash} | {len(fixture.catalog)} courses | "
          f"{len(fixture.requirement_groups)} requirement groups | "
          f"{len(fixture.scenarios)} synthesized students | {edges} prerequisite edges")
    # Internal consistency: does every code referenced anywhere actually exist?
    known = set(fixture.by_code)
    problems = []
    # A PROBLEM, not a note. Zero edges is a legal, silent state everywhere downstream — no
    # course has a `prereq_groups` field, so `prereq_violation` can never fire and PREREQUISITES
    # COME FIRST (the prompt's own top rule, and the largest Mode B failure class in every sweep
    # recorded so far) is unmeasurable. The models are still asked to obey it. See
    # `real_db._groups_from_tree` for where prerequisites are read from and
    # `catalog-ingest import-prerequisites` for how they get there.
    if not edges:
        problems.append(
            "this program exports ZERO prerequisite edges — no prerequisite data reached the "
            "database (neither `advisor.course_prerequisites` nor `prerequisite_rules` has "
            "rows for these courses). Prerequisite ordering cannot be scored in any mode, "
            "while every prompt still states it as a hard rule. Import them with "
            "`catalog-ingest import-prerequisites --from-dir <saved Banner pages>`."
        )
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
    for group in fixture.requirement_groups:
        for code in group.get("courses", []):
            if code not in known:
                problems.append(f"requirement group {group['id']} lists unknown course {code}")

    # `equivalent_to` is load-bearing for three separate checks (duplicates, prereqs,
    # coverage), and every way it can be wrong is silent at runtime.
    grouped = {c for g in fixture.requirement_groups for c in g.get("courses", [])}
    aliased_in_groups: list[str] = []
    for course in fixture.catalog:
        if not course.equivalent_to:
            continue
        if course.equivalent_to not in known:
            problems.append(
                f"{course.code} is equivalent_to {course.equivalent_to}, which is not in the "
                f"catalog — the substitution would silently never apply."
            )
        if course.code in grouped:
            # A NOTE, NOT A PROBLEM, since the catalog itself is the source. A hand-authored
            # fixture listing both a substitute and its primary in one group was an authoring
            # error; the real catalog genuinely offers both ("MA 16100 or MA 16500" is one
            # requirement stated as two options), and every scorer canonicalizes before
            # counting. What it still costs is Mode A: `select_remaining_courses` does not
            # canonicalize, so the deterministic planner can schedule both halves of a pair.
            aliased_in_groups.append(f"{course.code}->{course.equivalent_to}")
        if fixture.canonical.get(course.code) == course.code:
            problems.append(
                f"{course.code}'s equivalence resolves back to itself — a cycle in "
                f"`equivalent_to`. Substitution is silently a no-op for it."
            )
    for scenario in fixture.scenarios:
        for code in scenario.profile.completed_courses:
            if code not in known:
                problems.append(f"scenario {scenario.id} completed unknown course {code}")

    if aliased_in_groups:
        print(f"\nnote: {len(aliased_in_groups)} approved substitute(s) are listed in a "
              f"requirement group alongside their primary ({', '.join(aliased_in_groups[:6])}"
              f"{', ...' if len(aliased_in_groups) > 6 else ''}). Scored correctly (every "
              f"scorer canonicalizes); Mode A may schedule both halves of a pair.")

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
    # PER REQUEST, and not divided by the concurrency under EITHER engine — but for different
    # reasons, and the line says which. vLLM pages KV, so the window is simply free of
    # concurrency. llama.cpp buys the same guarantee by pre-allocating `num_ctx * parallel`,
    # which is why the resolved parallel (see `effective_parallel`) is what gets printed: it is
    # the number that had to fit in VRAM, and it may be lower than `run.parallel`.
    from harness.server import effective_parallel, engine  # noqa: PLC0415
    eng = engine(cfg)
    par = effective_parallel(cfg)
    if eng == "llamacpp":
        how = (f"{par} slot(s) pre-allocated as -c {cfg['run']['num_ctx'] * par}"
               + ("" if par == cfg["run"].get("parallel", 1)
                  else f", trimmed from run.parallel {cfg['run'].get('parallel', 1)}"))
    else:
        how = f"max_num_seqs {par} does not divide it under paged KV"
    print(f"\nper-request context: {slot} tokens (num_ctx {cfg['run']['num_ctx']}; {how}), "
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
    # Does ONE request still fit? Under vLLM this is purely a `num_ctx` question — paged KV
    # means concurrency no longer divides the window the way llama.cpp's `--parallel` did —
    # but the check is still worth running: the Mode B prompt grows with the program, and a
    # dual-major scenario has overrun the context before.
    slot_problem = check_slot_context(cfg["run"], approx)
    if slot_problem:
        problems.append(slot_problem)

    # Two entries pointing at the same checkpoint AND CONFIGURED IDENTICALLY benchmark one
    # model under several names and report a fake head-to-head. The report's duplicate-NAME
    # guard cannot see that: the names differ. It is an easy copy-paste mistake — and it has
    # already happened once in this file.
    #
    # SHARING A CHECKPOINT IS LEGITIMATE WHEN THE ENTRIES DIFFER, and since 2026-08-25 that is
    # how the advising-reasoning comparison is expressed: `qwen3.8-27b` and
    # `qwen3.8-27b-thinking` are the same weights with `advising_think` flipped, so ONE
    # `run.py run` produces both arms and every report table shows them as adjacent rows. So
    # the check is for entries that share a checkpoint and have nothing else to tell them
    # apart, which is the actual mistake.
    VARIANT_KEYS = ("advising_think", "think", "speculative", "draft_model", "kv_cache_dtype",
                    "cpu_offload_gb", "extra_args", "reasoning_parser", "tool_call_parser")
    by_path: dict[str, list[dict]] = {}
    for model in cfg["models"]:
        if not model.get("model_path"):
            continue
        by_path.setdefault(model["model_path"], []).append(model)
    for model_path, entries in by_path.items():
        if len(entries) < 2:
            continue
        seen: dict[tuple, str] = {}
        for m in entries:
            key = tuple(repr(m.get(k)) for k in VARIANT_KEYS)
            if key in seen:
                problems.append(
                    f"models {seen[key]} and {m['name']} share checkpoint {model_path} and "
                    f"differ in none of {', '.join(VARIANT_KEYS)} — they would benchmark one "
                    f"model under two names and fabricate a head-to-head."
                )
            seen[key] = m["name"]

    # A vLLM MODEL IS A DIRECTORY, not a file — config.json plus safetensors shards plus a
    # tokenizer — so "does it exist" is three questions, and the two that are not the
    # directory itself are the ones that produce a confusing failure. A directory holding only
    # the small files (a download that was interrupted after the metadata, which is the order
    # `hf download` fetches them in) starts a server that dies minutes later on a missing
    # tensor; a directory with no config.json fails inside transformers with an error about
    # the model TYPE rather than about the path.
    print("\nmodel checkpoints:")
    from harness.server import backend_cfg, engine, model_dir  # noqa: PLC0415
    eng = engine(cfg)
    root = Path(backend_cfg(cfg)["models_root"])
    missing = 0
    for model in cfg["models"]:
        # WHAT COUNTS AS PRESENT DEPENDS ON THE ENGINE: llama.cpp wants one .gguf FILE, vLLM a
        # checkpoint DIRECTORY. Checking the wrong one reports every model missing.
        if eng == "llamacpp":
            if not model.get("model_path"):
                missing += 1
                print(f"  NO PATH  {model['name']}: no `model_path:` in config.yaml "
                      f"(cannot run under --engine llamacpp)")
                continue
            path = model_dir(cfg, model)
            if not path.is_file():
                missing += 1
                print(f"  MISSING  {model['name']}: {path}")
                continue
            # A SHARDED GGUF IS ONLY AS PRESENT AS ITS LAST SHARD. `gguf:` names shard 1 and
            # llama.cpp resolves the rest by filename, so statting the named file alone reports
            # a 72.5 GB model as "ok, 0.0 GB" while it is still downloading — the check would
            # greenlight exactly the state it exists to catch. Count the set instead, and say
            # so when it is short.
            shards, expected = [path], 1
            m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
            if m:
                expected = int(m.group(2))
                stem = path.name[: m.start()]
                shards = sorted(path.parent.glob(f"{stem}-*-of-{m.group(2)}.gguf"))
            size_gb = sum(f.stat().st_size for f in shards) / 1e9
            if len(shards) < expected:
                missing += 1
                print(f"  PARTIAL  {model['name']}: {len(shards)}/{expected} shards, "
                      f"{size_gb:.1f} GB so far")
                continue
            label = f"{path.name}" if expected == 1 else f"{expected} shards"
            print(f"  ok       {model['name']}: {label} ({size_gb:.1f} GB)")
            continue
        if not model.get("model_path_vllm"):
            print(f"  n/a      {model['name']}: GGUF-only entry, no vLLM checkpoint "
                  f"(run it with --engine llamacpp)")
            continue
        path = root / model["model_path_vllm"]
        if not path.is_dir():
            missing += 1
            print(f"  MISSING  {model['name']}: {path}")
            continue
        if not (path / "config.json").exists():
            missing += 1
            problems.append(
                f"model {model['name']} points at {path}, which has no config.json — that is "
                f"a directory, but not a Hugging Face checkpoint."
            )
            print(f"  NO CONFIG {model['name']}: {path}")
            continue
        shards = list(path.glob("*.safetensors"))
        gb = sum(f.stat().st_size for f in shards) / 1e9
        if not shards:
            missing += 1
            problems.append(
                f"model {model['name']} at {path} has a config.json but no .safetensors "
                f"shards — an interrupted download looks exactly like this, and the server "
                f"will not fail until it is several minutes into loading."
            )
            print(f"  NO WEIGHTS {model['name']}: {path}")
            continue
        quant = "?"
        try:
            import json as _json
            quant = ((_json.loads((path / "config.json").read_text())
                      .get("quantization_config") or {}).get("quant_method") or "unquantized")
        except (OSError, ValueError):
            pass
        print(f"  {model['name']:16s} {len(shards)} shard(s), {gb:5.1f} GB, {quant}")
    print(f"  {len(cfg['models']) - missing}/{len(cfg['models'])} checkpoints present")

    # WILL THE BIGGEST ONE EVEN FIT? Weights plus KV have to live inside
    # `gpu_memory_utilization` of the card, and the failure mode when they nearly do is not a
    # crash — vLLM sizes a tiny KV pool and serves one sequence at a time, which reads as a
    # slow model rather than a misconfiguration.
    gpu_total = gpu_total_mb()
    if gpu_total:
        # THE BUDGET IS AN ENGINE-SPECIFIC NUMBER, and using vLLM's under llama.cpp produced
        # four false alarms: it flagged qwen3.8-27b's 23.1 GB gguf against a "22.1 GB budget"
        # and predicted concurrency capped at one request, while the model MEASURABLY loads at
        # 22,322 MiB and serves three slots. vLLM reserves `gpu_memory_utilization` of the card
        # up front and sizes its KV pool inside that; llama.cpp allocates weights and KV as it
        # needs them and simply fails if they do not fit, so the whole card is the budget, less
        # what the layers it spills to CPU do not occupy.
        # ⚠ GiB THROUGHOUT. `gpu_total_mb()` returns MiB, so `/1024` is GiB — but file sizes
        # were being divided by 1e9, which is decimal GB, and the two were compared directly.
        # That is a 7.4% error in the direction of false alarms: qwen3.8-27b's gguf is 23.1 GB
        # = 21.5 GiB, and against a "24.0" that was really 23.99 GiB it was reported as nearly
        # overflowing a card it MEASURABLY loads into at 22,322 MiB with room for three slots.
        if eng == "llamacpp":
            budget_gib = gpu_total / 1024
        else:
            budget_gib = gpu_total / 1024 * float(cfg["vllm"].get("gpu_memory_utilization", 0.92))
        for model in cfg["models"]:
            if eng == "llamacpp":
                if not model.get("model_path"):
                    continue
                path = model_dir(cfg, model)
                if not path.is_file():
                    continue
                weights_gib = path.stat().st_size / 2**30
                # SHARDS COUNT TOGETHER, and offloaded experts do not count at all: an entry
                # with `n_cpu_moe` deliberately keeps most of its weights in system RAM, so
                # comparing the full checkpoint size against VRAM would flag exactly the models
                # that are configured correctly. qwen3.8-flash-next is 72.5 GB on disk and
                # occupies 21.5 GB of card.
                m_sh = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
                if m_sh:
                    stem = path.name[: m_sh.start()]
                    weights_gib = sum(f.stat().st_size for f in
                                      path.parent.glob(f"{stem}-*-of-{m_sh.group(2)}.gguf")) / 2**30
                if model.get("n_cpu_moe"):
                    continue
            else:
                if not model.get("model_path_vllm"):
                    continue
                path = root / model["model_path_vllm"]
                if not path.is_dir():
                    continue
                weights_gib = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 2**30
            if weights_gib and weights_gib > budget_gib * 0.9:
                problems.append(
                    f"model {model['name']} is ~{weights_gib:.1f} GiB of weights against a "
                    f"~{budget_gib:.1f} GiB budget on a {gpu_total / 1024:.1f} GiB card"
                    + ("" if eng == "llamacpp" else
                       f" (gpu_memory_utilization "
                       f"{cfg['vllm'].get('gpu_memory_utilization', 0.92)})")
                    + f". What is left for the KV cache will cap concurrency; "
                      f"check `kv_cache` in the run's meta_*.json before believing any "
                      f"throughput number from it."
                )


    problems += _check_draft_models(cfg, root)

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
    """Launch the vLLM server for one model with the harness's exact settings and hold it.

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
    print(f"vLLM up: {handle.base_url}")
    print(f"  argv: {' '.join(handle.argv)}")
    print(f"  log:  {handle.log_path}")
    print(f"  kv cache: {handle.kv_cache()}")
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

    from harness.real_db import fixture_from_database, load_real_db  # noqa: PLC0415
    from harness.server import slot_context  # noqa: PLC0415

    cfg = _cfg()
    real_db_cfg = cfg["real_db"]
    fixture = fixture_from_database(load_real_db(
        os.environ.get("MODEL_EVAL_REAL_DB_URL", real_db_cfg["url"]),
        os.environ.get("MODEL_EVAL_REAL_DB_PROGRAM_ID", real_db_cfg["program_id"]),
        budget_tokens=slot_context(cfg["run"]) - cfg["run"]["max_plan_tokens"],
    ))
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


if __name__ == "__main__":
    main()
