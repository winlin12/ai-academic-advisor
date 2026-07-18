"""Report generation: per-model table, head-to-head (best 8GB vs reference), and the
mitigation before/after comparison — the table the purchase decision actually reads.

No composite score. Metrics are reported side by side with run-to-run variance
(the min–max spread of each rate when recomputed per run replicate) and the report
refuses to bless gaps that the pre-registered decision rule can't distinguish from
noise at the current sample size.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _rate(vals: list[bool]) -> float | None:
    return sum(vals) / len(vals) if vals else None


class ModelStats:
    def __init__(self) -> None:
        self.sql_records: list[dict] = []
        self.summary_records: list[dict] = []
        self.e2e_records: list[dict] = []
        self.env: dict[str, Any] = {}

    # -- helpers -------------------------------------------------------------
    def _by_run(self, key) -> list[float]:
        """Recompute a rate per run_idx replicate — the honest variance display."""
        buckets: dict[int, list[bool]] = defaultdict(list)
        for r in self.sql_records:
            v = key(r)
            if v is not None:
                buckets[r["run_idx"]].append(v)
        return [_rate(v) for v in buckets.values() if v]

    def _overall(self, key) -> float | None:
        vals = [key(r) for r in self.sql_records if key(r) is not None]
        return _rate(vals) if vals else None

    @staticmethod
    def _fmt(rate: float | None, spread: list[float]) -> str:
        if rate is None:
            return "—"
        if len(spread) > 1:
            return f"{_pct(rate)} ({_pct(min(spread))}–{_pct(max(spread))})"
        return _pct(rate)

    # -- metric keys -----------------------------------------------------------
    @staticmethod
    def k_valid(r):    # answerable questions only: did emitted-or-not SQL execute?
        return r["sql_valid"] if r["expected_behavior"] == "answer" else None

    @staticmethod
    def k_correct(r):
        return r["sql_correct"]  # None when no gold or no SQL emitted

    @staticmethod
    def k_decline(r):
        if r["category"] in ("out_of_scope", "adversarial") and r["expected_behavior"] == "decline":
            return r["behavior_ok"]
        return None

    @staticmethod
    def k_clarify(r):
        return r["behavior_ok"] if r["expected_behavior"] == "clarify" else None

    def correct_honest_value(self, gold_ids: set[str]) -> float | None:
        vals = [bool(r["sql_correct"]) for r in self.sql_records if r["question_id"] in gold_ids]
        return _rate(vals) if vals else None

    def correct_honest(self, gold_ids: set[str]) -> str:
        """SQL_CORRECT with abstentions counted as misses, not excluded. k_correct's
        denominator is only rows where the model actually emitted SQL against a gold
        question — a decline/clarify/unparseable on a real question silently vanishes
        from that rate instead of counting against it. This recomputes over every
        gold-eligible call: r['sql_correct'] is True -> pass, anything else -> fail."""
        def key(r):
            return bool(r["sql_correct"]) if r["question_id"] in gold_ids else None
        buckets: dict[int, list[bool]] = defaultdict(list)
        vals: list[bool] = []
        for r in self.sql_records:
            v = key(r)
            if v is None:
                continue
            vals.append(v)
            buckets[r["run_idx"]].append(v)
        if not vals:
            return "—"
        spread = [_rate(v) for v in buckets.values() if v]
        return self._fmt(_rate(vals), spread)

    def e2e_rate(self) -> str:
        """Heuristic pass rate for the end-to-end stage (model's OWN retrieval, not gold
        rows, fed to the summarizer). Not a verdict — see the honesty ledger in scorers.py."""
        if not self.e2e_records:
            return "—"
        passed = sum(1 for r in self.e2e_records if r.get("e2e_auto_pass"))
        return f"{passed}/{len(self.e2e_records)} heuristic-pass"

    def metric(self, key) -> str:
        return self._fmt(self._overall(key), self._by_run(key))

    def metric_value(self, key) -> float | None:
        return self._overall(key)

    def latency(self) -> str:
        ts = sorted(r["total_s"] for r in self.sql_records if r.get("total_s"))
        if not ts:
            return "—"
        p50 = statistics.median(ts)
        p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
        return f"{p50:.1f}s / {p95:.1f}s"

    def flagged_summaries(self) -> str:
        if not self.summary_records:
            return "—"
        flagged = sum(1 for r in self.summary_records if r.get("faithfulness_flags"))
        return f"{flagged}/{len(self.summary_records)} flagged (heuristic; grade manually)"

    def consistency(self) -> str:
        """Fraction of questions where all N runs produced the same behavior_ok outcome."""
        buckets: dict[str, set] = defaultdict(set)
        for r in self.sql_records:
            buckets[r["question_id"]].add(r["behavior_ok"])
        if not buckets:
            return "—"
        stable = sum(1 for v in buckets.values() if len(v) == 1)
        return f"{stable}/{len(buckets)} questions run-stable"


def load_stats(path: Path) -> dict[str, ModelStats]:
    stats: dict[str, ModelStats] = defaultdict(ModelStats)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        s = stats[r["model"]]
        if r.get("stage") == "env":
            s.env = r
        elif r.get("stage") == "sql" and "error" not in r:
            s.sql_records.append(r)
        elif r.get("stage") == "summary" and "error" not in r:
            s.summary_records.append(r)
        elif r.get("stage") == "e2e" and "error" not in r:
            s.e2e_records.append(r)
    return dict(stats)


def gold_empty_questions(root: Path, cfg: dict, questions: list[dict]) -> list[str]:
    """Gold SQL that returns zero rows against the actual seed DB is a trap: ANY candidate
    query that also returns nothing — right or wrong — trivially matches it, inflating
    SQL_CORRECT for whichever model happens to give up gracefully. Flag these so they get
    fixed (add seed rows or drop the gold) instead of silently padding scores."""
    db_file = root / cfg["paths"]["db_file"]
    if not db_file.exists():
        return []
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    empty: list[str] = []
    for q in questions:
        gold = (q.get("gold_sql") or "").strip()
        if not gold:
            continue
        try:
            rows = conn.execute(gold.rstrip(";")).fetchall()
        except sqlite3.Error:
            continue
        if not rows:
            empty.append(q["id"])
    conn.close()
    return empty


def _model_table(stats: dict[str, ModelStats], cfg: dict, gold_ids: set[str]) -> list[str]:
    bracket = {m["name"]: m.get("bracket", "?") for m in cfg["models"]}
    lines = [
        "| Model | Bracket | SQL_VALID | SQL_CORRECT (attempted) | SQL_CORRECT (honest) "
        "| E2E (heuristic) | DECLINE | CLARIFY | Faithfulness | p50/p95 | Consistency | Offload |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def sort_key(item):
        return -(item[1].correct_honest_value(gold_ids) or 0)
    for name, s in sorted(stats.items(), key=sort_key):
        off = (s.env.get("offload") or {})
        off_s = off.get("layers") or ("full" if off.get("fully_offloaded") else "UNVERIFIED")
        lines.append(
            f"| {name} | {bracket.get(name, '?')} | {s.metric(ModelStats.k_valid)} "
            f"| {s.metric(ModelStats.k_correct)} | {s.correct_honest(gold_ids)} "
            f"| {s.e2e_rate()} | {s.metric(ModelStats.k_decline)} "
            f"| {s.metric(ModelStats.k_clarify)} | {s.flagged_summaries()} "
            f"| {s.latency()} | {s.consistency()} | {off_s} |"
        )
    return lines


def best_8gb(stats: dict[str, ModelStats], cfg: dict) -> str | None:
    names = [m["name"] for m in cfg["models"] if m.get("bracket") == "8gb"]
    ranked = sorted(
        (n for n in names if n in stats and stats[n].sql_records),
        key=lambda n: (
            stats[n].metric_value(ModelStats.k_correct) or 0,
            stats[n].metric_value(ModelStats.k_valid) or 0,
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _head_to_head(base: dict[str, ModelStats], mitig: dict[str, ModelStats],
                  cfg: dict, gold_ids: set[str]) -> list[str]:
    ref_name = next((m["name"] for m in cfg["models"] if m.get("bracket") == "reference"), None)
    champ = best_8gb(base, cfg)
    if not champ or not ref_name or ref_name not in base:
        return ["*(head-to-head unavailable: need results for both a 8gb model and "
                f"the reference model {ref_name!r})*"]

    ref, b8 = base[ref_name], base[champ]
    m8 = mitig.get(champ)
    lines = [
        f"Best 8GB model by SQL_CORRECT: **{champ}**",
        "",
        f"| Metric | {champ} (baseline) | {champ} (mitigated) | {ref_name} (reference) |",
        "|---|---|---|---|",
    ]
    for label, key in [("SQL_VALID", ModelStats.k_valid),
                       ("SQL_CORRECT (attempted)", ModelStats.k_correct),
                       ("DECLINE", ModelStats.k_decline),
                       ("CLARIFY", ModelStats.k_clarify)]:
        lines.append(f"| {label} | {b8.metric(key)} | "
                     f"{m8.metric(key) if m8 else 'not run'} | {ref.metric(key)} |")
    lines += [
        f"| SQL_CORRECT (honest, abstentions=miss) | {b8.correct_honest(gold_ids)} | "
        f"{m8.correct_honest(gold_ids) if m8 else 'not run'} | {ref.correct_honest(gold_ids)} |",
        f"| E2E (heuristic pass) | {b8.e2e_rate()} | "
        f"{m8.e2e_rate() if m8 else 'not run'} | {ref.e2e_rate()} |",
        f"| Faithfulness (heuristic triage) | {b8.flagged_summaries()} | "
        f"{m8.flagged_summaries() if m8 else 'not run'} | {ref.flagged_summaries()} |",
        f"| Latency p50/p95 | {b8.latency()} | {m8.latency() if m8 else 'not run'} "
        f"| {ref.latency()} (5070 Ti — says NOTHING about 3090 latency) |",
    ]

    # Mechanical evaluation of the pre-registered rule.
    dec = cfg.get("decision", {})
    golded = len({r["question_id"] for r in b8.sql_records if r["sql_correct"] is not None})
    lines += ["", "### Decision rule check", ""]
    if golded < dec.get("min_gold_questions", 30):
        lines.append(
            f"⚠️ Only **{golded}** gold-scored questions (< {dec.get('min_gold_questions', 30)} "
            "pre-registered minimum). Any gap below ~18pp is within binomial noise at this "
            "sample size — **no purchase decision can be read from this run.**")
    subject = m8 or b8
    subj_label = "mitigated" if m8 else "BASELINE (mitigation not run yet — run it first)"
    c8 = subject.metric_value(ModelStats.k_correct)
    cref = ref.metric_value(ModelStats.k_correct)
    if c8 is not None and cref is not None:
        gap = (cref - c8) * 100
        need_gap = dec.get("sql_correct_gap_pp", 15)
        floor = dec.get("sql_correct_8gb_floor", 0.80)
        lines.append(f"- 8GB ({subj_label}) SQL_CORRECT = {_pct(c8)}; "
                     f"{ref_name} = {_pct(cref)}; gap = {gap:.0f}pp "
                     f"(threshold: ≥{need_gap}pp AND 8GB < {_pct(floor)})")
        if gap >= need_gap and c8 < floor and golded >= dec.get("min_gold_questions", 30):
            lines.append("- **Rule fires: the data supports buying the 3090** "
                         "(subject to your manual faithfulness grades).")
        else:
            lines.append("- **Rule does not fire: default NO holds — keep the 2060 Super.**")
    return lines


def generate_report(root: Path) -> Path:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    results_dir = root / cfg["paths"]["results_dir"]
    base = load_stats(results_dir / "runs_baseline.jsonl")
    mitig = load_stats(results_dir / "runs_mitigated.jsonl")
    if not base:
        raise SystemExit("No baseline results found — run `python run.py run` first.")

    questions = yaml.safe_load((root / cfg["paths"]["questions"]).read_text())
    gold_ids = {q["id"] for q in questions if (q.get("gold_sql") or "").strip()}

    meta_path = results_dir / "meta_baseline.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    lines = ["# Model Evaluation Report", ""]
    if meta:
        lines += [f"Ollama version: `{meta.get('ollama_version')}` · options: "
                  f"`{meta.get('options')}` · runs/pair: {meta.get('runs_per_pair')} · "
                  f"warmup discarded: {meta.get('warmup_discard')}", ""]
    hashes = {r["static_hash"] for s in base.values() for r in s.sql_records}
    if len(hashes) > 1:
        lines += ["⚠️ **MULTIPLE STATIC-PROMPT HASHES IN ONE RESULT FILE** — someone edited "
                  "schema.sql or prompts.py mid-run. These rows are not comparable. "
                  "Delete results/ and re-run.", ""]

    empty_gold = gold_empty_questions(root, cfg, questions)
    if empty_gold:
        lines += [f"⚠️ **{len(empty_gold)} gold quer{'y' if len(empty_gold)==1 else 'ies'} "
                  f"return ZERO rows against the seed DB**: {', '.join(empty_gold)}. Any "
                  "candidate SQL that also returns nothing trivially matches these and counts "
                  "as SQL_CORRECT — add the missing seed rows or drop the gold.", ""]

    dupes = [m["name"] for m in cfg["models"]
             if [x["name"] for x in cfg["models"]].count(m["name"]) > 1]
    if dupes:
        lines += [f"⚠️ **Model(s) listed more than once in config.yaml**: "
                  f"{', '.join(sorted(set(dupes)))}. Each duplicate runs separately and their "
                  "records get pooled under one name here — if the duplicates differ in "
                  "`think` or any other option, this row is a blend of two conditions. "
                  "Remove the duplicate entries.", ""]

    lines += ["## Per-model results", "",
              "Rates show overall (min–max across run replicates). SQL_CORRECT (attempted) "
              "covers only calls where the model emitted SQL against a gold question — a "
              "decline/clarify/unparseable silently drops out of that denominator. "
              "SQL_CORRECT (honest) counts every gold-eligible call, scoring abstentions as "
              "misses. E2E feeds the model's OWN retrieved rows (not gold rows) into the "
              "summarizer — closer to what a student actually sees — and is a heuristic "
              "pass/fail triage, not a verdict. Faithfulness numbers are heuristic triage "
              "counts — the real grade is your manual review of `results/review_queue.jsonl`.",
              ""]
    lines += _model_table(base, cfg, gold_ids)
    lines += ["", "## Head-to-head: best 8GB vs reference", ""]
    lines += _head_to_head(base, mitig, cfg, gold_ids)

    # Manual review queue: every summary + e2e answer + flagged/unparseable behavior call.
    queue = results_dir / "review_queue.jsonl"
    with queue.open("w") as f:
        for stats in list(base.values()) + list(mitig.values()):
            for r in stats.summary_records:
                f.write(json.dumps({k: r[k] for k in
                        ("model", "question_id", "run_idx", "mitigated", "output",
                         "rows_json", "faithfulness_flags")}) + "\n")
            for r in stats.e2e_records:
                if not r.get("needs_review"):
                    continue
                f.write(json.dumps({k: r.get(k) for k in
                        ("model", "question_id", "run_idx", "mitigated", "output",
                         "retrieval_correct", "rows_json", "gold_rows_json",
                         "faithfulness_flags", "recall_flags", "e2e_auto_pass")}) + "\n")
            for r in stats.sql_records:
                if r.get("needs_review"):
                    f.write(json.dumps({k: r.get(k) for k in
                            ("model", "question_id", "run_idx", "mitigated", "category",
                             "expected_behavior", "action", "output")}) + "\n")
    lines += ["", f"Manual review queue written to `{queue}` — faithfulness, e2e, and "
              "off-format decline calls are graded by you, not by this script.", ""]

    out = results_dir / "report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    return out
