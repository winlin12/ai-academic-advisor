#!/usr/bin/env python3
"""Evaluation harness CLI. Standalone: needs Python 3.10+, PyYAML, and a reachable
Ollama server. Runs entirely from this directory; never touches the app's backend.

  python run.py db                          rebuild eval.sqlite from schema.sql
  python run.py check                       env sanity: Ollama version, GPU, questions
  python run.py run [--models a,b] [--brackets 8gb,small]
  python run.py run --mitigate --models X   mitigation pass (retry + grammar + few-shot v2)
  python run.py report                      per-model table + head-to-head + decision rule
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("db", help="build eval.sqlite from schema.sql")
    sub.add_parser("check", help="sanity-check environment and question set")
    sub.add_parser("report", help="generate results/report.md")

    runp = sub.add_parser("run", help="execute the evaluation")
    runp.add_argument("--models", help="comma-separated model names (default: all)")
    runp.add_argument("--brackets", help="comma-separated brackets: 8gb,small,reference")
    runp.add_argument("--mitigate", action="store_true",
                      help="mitigation mode: few-shot v2 + JSON-schema output + one "
                           "retry-on-invalid-SQL; writes runs_mitigated.jsonl")

    args = ap.parse_args()

    if args.cmd == "db":
        from harness.db import build_db
        build_db(ROOT / "schema.sql", ROOT / "eval.sqlite")
        print("Built eval.sqlite (read-write copy is discarded; runs open it read-only).")
    elif args.cmd == "check":
        check(ROOT)
    elif args.cmd == "run":
        from harness.runner import run_eval
        run_eval(
            ROOT,
            models=args.models.split(",") if args.models else None,
            brackets=args.brackets.split(",") if args.brackets else None,
            mitigate=args.mitigate,
        )
    elif args.cmd == "report":
        from harness.report import generate_report
        generate_report(ROOT)


def check(root: Path) -> None:
    """Everything that can be verified without spending a single generation."""
    import sqlite3

    import yaml

    from harness.db import build_db, execute_select, ro_connect
    from harness.ollama_client import OllamaClient, OllamaError
    from harness.prompts import PromptBuilder
    from harness.runner import gpu_memory_used_mb

    cfg = yaml.safe_load((root / "config.yaml").read_text())
    questions = yaml.safe_load((root / cfg["paths"]["questions"]).read_text())

    build_db(root / "schema.sql", root / cfg["paths"]["db_file"])
    conn = ro_connect(root / cfg["paths"]["db_file"])
    bad, golds = [], 0
    for q in questions:
        gold = (q.get("gold_sql") or "").strip()
        if gold:
            golds += 1
            try:
                cols, rows = execute_select(conn, gold)
                if not rows:
                    bad.append(f"{q['id']}: gold SQL returned 0 rows (seed data gap?)")
            except sqlite3.Error as exc:
                bad.append(f"{q['id']}: gold SQL error: {exc}")
    print(f"questions: {len(questions)} total, {golds} with gold SQL "
          f"({len(questions) - golds} golds still to write)")
    for b in bad:
        print(f"  GOLD PROBLEM — {b}")
    print(f"static prompt hash (v1): {PromptBuilder(root / 'schema.sql').build_sql_prompt('x')[1]}")
    print(f"nvidia-smi: {'available ' + str(gpu_memory_used_mb()) + ' MB used' if gpu_memory_used_mb() is not None else 'NOT available (fine on the Mac; required on the GPU boxes)'}")
    try:
        v = OllamaClient(cfg["ollama"]["base_url"], timeout_s=5).version()
        print(f"ollama: reachable, version {v}")
    except OllamaError as exc:
        print(f"ollama: {exc}")
    if not cfg["ollama"].get("server_log_cmd"):
        print("server_log_cmd: not set — GPU layer offload will be UNVERIFIED in reports "
              "(set it to e.g. 'journalctl -u ollama -n 4000 --no-pager' on the server)")


if __name__ == "__main__":
    main()
