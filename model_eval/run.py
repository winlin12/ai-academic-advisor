#!/usr/bin/env python3
"""Evaluation harness CLI. Standalone: Python 3.10+ and PyYAML, nothing from the app.

  python run.py doctor                      diagnose WSL -> Windows llama-server reachability
  python run.py check                       validate fixture, prompts, GPU, model files
  python run.py serve <model>               launch llama-server for one model and hold it
  python run.py run [--models a,b] [--brackets 8gb,coder] [--tasks plan_b,qa]
  python run.py run --mitigate --models X   mitigation pass (multi-iteration revise loop)
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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="diagnose WSL -> Windows llama-server networking")
    sub.add_parser("check", help="validate fixture, prompts, GPU, and model files")
    sub.add_parser("report", help="generate results/report.md")
    sub.add_parser("parity", help="check the vendored planner against the app's")
    sub.add_parser("fixture-check", help="diff the plan fixture against the catalog DB")
    sub.add_parser("refresh-offerings",
                   help="rewrite the fixture's offered_terms from observed PurdueIO offerings")

    servep = sub.add_parser("serve", help="launch llama-server for one model and hold it")
    servep.add_argument("model", help="model name from config.yaml")

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
    elif args.cmd == "report":
        from harness.report import generate_report
        generate_report(ROOT)
    elif args.cmd == "parity":
        parity()
    elif args.cmd == "fixture-check":
        fixture_check()
    elif args.cmd == "refresh-offerings":
        refresh_offerings()


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


# --- doctor ------------------------------------------------------------------------------------


def doctor() -> None:
    """The single most likely reason a run fails on this box, diagnosed explicitly.

    llama-server is a native Windows process; the harness runs in WSL2 NAT mode. WSL cannot
    reach the Windows loopback, so the server binds 0.0.0.0 and is reached at the default
    gateway — a path Windows Firewall blocks by default. Left undiagnosed this looks exactly
    like "the model failed to load", which sends you debugging the wrong thing.
    """
    from harness.server import gpu_memory_used_mb, windows_host_ip

    cfg = _cfg()
    port = cfg["llamacpp"]["port"]
    print("=== WSL -> Windows llama-server reachability ===\n")

    host = windows_host_ip()
    print(f"Windows host (WSL default gateway): {host or 'NOT FOUND'}")
    if not host:
        print("  Could not read a default route. Set llamacpp.host explicitly in config.yaml.")
        return

    exe = Path(cfg["llamacpp"]["server_exe"].replace("\\", "/").replace("D:", "/mnt/d"))
    print(f"llama-server.exe: {'found' if exe.exists() else 'NOT FOUND at ' + str(exe)}")

    gpu = gpu_memory_used_mb()
    print(f"nvidia-smi: {'available, ' + str(gpu) + ' MB used' if gpu else 'NOT available'}")

    listening = _port_open(host, port, timeout=2)
    print(f"\n{host}:{port} -> {'OPEN' if listening else 'closed'}")

    if listening:
        print("\nA server is already reachable on that port. If it is not the model you want to "
              "test, stop it — `run.py run` will start its own.")
        return

    print("\nNo server is listening there right now, which is expected when one isn't running.")
    print("Testing whether WSL can reach the Windows host AT ALL (independent of llama.cpp):\n")

    probe_port = 8123
    proc = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command",
         f"$l=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any,{probe_port});"
         f"$l.Start(); Start-Sleep -Seconds 10; $l.Stop()"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        import time
        time.sleep(4)
        reachable = _port_open(host, probe_port, timeout=3)
    finally:
        proc.terminate()

    if reachable:
        print(f"  ✅ WSL reached the Windows host on :{probe_port}. Networking is fine — start a "
              f"model with `python run.py serve <model>` or just `python run.py run`.")
        return

    print(f"  ❌ WSL could NOT reach the Windows host on :{probe_port}, even though a listener "
          f"was running there.")
    print("\n     This is Windows Firewall blocking inbound connections from the WSL adapter.")
    print("     Fix it once, with one UAC click:\n")
    print("         powershell.exe -ExecutionPolicy Bypass -File setup/allow_wsl_llamacpp.ps1\n")
    print("     Alternative (survives WSL IP changes, but needs `wsl --shutdown`, which ends")
    print("     any running WSL session): add `networkingMode=mirrored` under [wsl2] in")
    print("     C:\\Users\\<you>\\.wslconfig, then set llamacpp.host to 127.0.0.1 here.")


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
    from harness.fixtures import load_fixture
    from harness.planner import generate_plan
    from harness.plan_scorers import score_plan, plan_to_semesters
    from harness.prompts import PromptBuilder
    from harness.server import gpu_memory_used_mb

    cfg = _cfg()
    fixture = load_fixture(ROOT / cfg["paths"]["plan_fixture"])
    prompts = PromptBuilder(fixture)

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
        if not course.offered_terms:
            problems.append(f"{course.code} has no offered_terms")
    for group in fixture.requirement_groups:
        for code in group.get("courses", []):
            if code not in known:
                problems.append(f"requirement group {group['id']} lists unknown course {code}")
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

    # A prompt that doesn't fit in num_ctx isn't a model failure; catch it here.
    system, user, _ = prompts.plan_freeform(scenario)
    approx = (len(system) + len(user)) // 4
    budget = cfg["run"]["num_ctx"] - cfg["run"]["max_plan_tokens"]
    print(f"\nlongest prompt (mode B): ~{approx} tokens vs. num_ctx {cfg['run']['num_ctx']} "
          f"minus max_plan_tokens {cfg['run']['max_plan_tokens']} = {budget} available")
    if approx > budget:
        problems.append(
            f"the Mode B prompt (~{approx} tok) does not leave room to generate within "
            f"num_ctx={cfg['run']['num_ctx']}. Raise num_ctx or trim the fixture catalog."
        )

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
    root = cfg["llamacpp"]["models_root"].replace("\\", "/").replace("D:", "/mnt/d")
    missing = 0
    for model in cfg["models"]:
        path = Path(root) / model["gguf"].replace("\\", "/")
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
    app_catalog = [
        AppCourse(
            code=c.code, title=c.title, credits=c.credits, prereqs=list(c.prereqs),
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
