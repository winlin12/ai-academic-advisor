"""Scoring. Honesty ledger — what each scorer really is:

  SQL_VALID     automatic (parse + execute on a read-only connection)
  SQL_CORRECT   automatic WHERE a gold query exists (execution-accuracy comparison)
  DECLINE/CLARIFY  automatic when the model follows the sentinel contract; off-format
                   outputs fall back to a phrase heuristic and are FLAGGED for manual
                   review — the flag count is reported, not hidden
  FAITHFULNESS  a heuristic entity check (course codes / numbers in the answer that are
                 absent from the retrieved rows) — it catches entity hallucination ONLY.
                 Relational hallucination ("X must come before Y") is NOT machine-checked
                 here and every summarization answer goes to the manual review queue.
                 Do not quote the heuristic as a faithfulness rate; quote your manual one.

No composite score exists anywhere in this harness, deliberately.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import db as dbmod

_SENTINEL_RE = re.compile(r"^\s*(SQL|CLARIFY|DECLINE)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:sql|json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)  # r1-style leakage
_DECLINE_PHRASES = (
    "i can only help", "i can't help", "i cannot help", "i'm not able", "i am not able",
    "outside", "don't have information", "do not have information", "cannot answer",
    "can't answer", "only assist", "i'm sorry", "i apologize",
)
_CLARIFY_MARKERS = ("which ", "what program", "could you clarify", "can you clarify", "do you mean")


@dataclass
class ParsedOutput:
    action: str            # 'sql' | 'clarify' | 'decline' | 'unparseable'
    sql: str | None = None
    via: str = "sentinel"  # 'sentinel' | 'json' | 'heuristic'
    needs_review: bool = False


def parse_output(text: str) -> ParsedOutput:
    """Recover the model's intended action. Sentinel first, then structured JSON, then
    heuristics (flagged). An unparseable answer is a real failure mode and stays one."""
    cleaned = _THINK_RE.sub("", text).strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    m = _SENTINEL_RE.match(cleaned)
    if m:
        kind = m.group(1).lower()
        body = m.group(2).strip()
        if kind == "sql":
            return ParsedOutput(action="sql", sql=body)
        return ParsedOutput(action=kind)

    try:  # structured (mitigation) mode
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and obj.get("action") in ("sql", "clarify", "decline"):
            return ParsedOutput(action=obj["action"], sql=obj.get("sql"), via="json")
    except (json.JSONDecodeError, TypeError):
        pass

    lowered = cleaned.lower()
    if lowered.lstrip().startswith(("select", "with")):
        return ParsedOutput(action="sql", sql=cleaned, via="heuristic", needs_review=True)
    if any(p in lowered for p in _DECLINE_PHRASES):
        return ParsedOutput(action="decline", via="heuristic", needs_review=True)
    if cleaned.endswith("?") and any(p in lowered for p in _CLARIFY_MARKERS):
        return ParsedOutput(action="clarify", via="heuristic", needs_review=True)
    return ParsedOutput(action="unparseable", needs_review=True)


@dataclass
class SqlScore:
    valid: bool | None = None      # None = model didn't emit SQL
    correct: bool | None = None    # None = no gold, or no SQL to check
    error: str | None = None
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


def score_sql(conn: sqlite3.Connection, sql: str | None, gold_sql: str | None) -> SqlScore:
    if not sql:
        return SqlScore()
    try:
        columns, rows = dbmod.execute_select(conn, sql)
    except sqlite3.Error as exc:
        return SqlScore(valid=False, error=str(exc))
    score = SqlScore(valid=True, rows=rows, columns=columns)
    if gold_sql:
        gold_cols, gold_rows = dbmod.execute_select(conn, gold_sql)
        score.correct = dbmod.rows_equal(rows, gold_rows)
    return score


def score_behavior(parsed: ParsedOutput, expected: str) -> bool:
    """expected: 'answer' | 'clarify' | 'decline'. For 'answer', emitting SQL is the
    behavior check (whether it's GOOD SQL is score_sql's job)."""
    return parsed.action == ("sql" if expected == "answer" else expected)


_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s?\d{3,5}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def faithfulness_flags(answer: str, rows_json: str) -> list[str]:
    """Entities in the answer unsupported by the retrieved rows. Heuristic ONLY —
    a triage filter for the manual review queue, not a verdict."""
    flags: list[str] = []
    haystack = rows_json.replace(" ", " ")
    for code in set(_COURSE_CODE_RE.findall(answer)):
        if code not in haystack and code.replace(" ", "") not in haystack.replace(" ", ""):
            flags.append(f"course code not in rows: {code}")
    codes_text = " ".join(_COURSE_CODE_RE.findall(answer))
    for num in set(_NUMBER_RE.findall(answer)):
        if num in codes_text:
            continue  # digits inside course codes are already handled above
        if num not in haystack:
            flags.append(f"number not in rows: {num}")
    return flags


def record_scores(record: dict[str, Any], parsed: ParsedOutput, sql_score: SqlScore,
                  expected: str) -> dict[str, Any]:
    """Flatten scoring into the result record (single place, so the JSONL schema is stable)."""
    record.update(
        action=parsed.action,
        parse_via=parsed.via,
        needs_review=parsed.needs_review,
        behavior_ok=score_behavior(parsed, expected),
        sql=parsed.sql,
        sql_valid=sql_score.valid,
        sql_correct=sql_score.correct,
        sql_error=sql_score.error,
    )
    return record
