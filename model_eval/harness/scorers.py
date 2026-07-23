"""Scoring for the free-text stages, plus structured-output extraction.

Honesty ledger — what each scorer really is:

  STRUCTURE_OK    automatic. Did grammar-constrained decoding actually yield JSON matching the
                  requested shape? Under llama.cpp's GBNF response_format this SHOULD be ~100%,
                  so anything below it is a real signal (a build/template mismatch, a truncated
                  generation, or a model that ignores the grammar by emitting a preamble).
  PLAN_VIABLE     automatic and decidable — see plan_scorers.py. The headline metric.
  FAITHFULNESS    a heuristic ENTITY check (course codes / numbers in the answer that are absent
                  from the retrieved context) — triage only. Relational hallucination
                  ("CS 38100 must come before CS 37300") is NOT machine-checked here, and every
                  free-text answer goes to the manual review queue. Do not quote the heuristic
                  as a faithfulness rate; quote your manual one.
  RECALL          the mirror image: context course codes that never appear in the answer.
                  Also triage.
  ABSTAIN         automatic-ish phrase heuristic for "I don't have that rule on file". The app's
                  QA prompt asks for exactly that behaviour when context is thin, so questions
                  with deliberately empty/irrelevant context are scored on it.

No composite score exists anywhere in this harness, deliberately.

The old SQL scorers (parse_output / score_sql / score_behavior) are gone: the app no longer
asks a model for SQL. Their execution-accuracy machinery went with them, along with schema.sql
and eval.sqlite.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s?\d{3,5}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_ABSTAIN_PHRASES = (
    "don't have", "do not have", "not in the context", "no rule on file",
    "isn't in the", "is not in the", "cannot answer", "can't answer",
    "confirm with", "check with", "contact your", "i'm not able", "i am not able",
    "i can only", "no information",
)


def strip_reasoning(text: str) -> str:
    """Mirrors ``llamacpp_client.strip_reasoning`` in the app. Reasoning is disabled at server
    launch (``--reasoning off``), so this should be a no-op — it stays because a leaked
    ``<think>`` block would otherwise be scored as hallucinated content."""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from a model response.

    Under grammar-constrained decoding the response IS a bare object, so the fence-stripping
    and brace-scanning fallbacks below should never fire. They exist so that a model which
    ignores the grammar is scored on the JSON it *meant* to emit rather than on the harness's
    strictness — a failure to follow the schema should show up as a bad plan, not as an
    unparseable one, unless it truly is unparseable.
    """
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    for candidate in (cleaned, _first_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_object(text: str) -> str | None:
    """The first balanced {...} span, so a preamble ("Here is the plan:") doesn't kill parsing."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def faithfulness_flags(answer: str, context: str) -> list[str]:
    """Entities in the answer unsupported by the retrieved context. Heuristic ONLY."""
    flags: list[str] = []
    for code in sorted(set(_COURSE_CODE_RE.findall(answer))):
        if code not in context and code.replace(" ", "") not in context.replace(" ", ""):
            flags.append(f"course code not in context: {code}")
    codes_text = " ".join(_COURSE_CODE_RE.findall(answer))
    for num in sorted(set(_NUMBER_RE.findall(answer))):
        if num in codes_text:
            continue  # digits inside course codes are handled above
        if num not in context:
            flags.append(f"number not in context: {num}")
    return flags


def recall_flags(answer: str, context: str) -> list[str]:
    """Course codes present in the context that never show up in the answer — the mirror image
    of faithfulness_flags (invention vs. omission). Scoped to course codes to avoid noise."""
    flags: list[str] = []
    for code in sorted(set(_COURSE_CODE_RE.findall(context))):
        if code not in answer and code.replace(" ", "") not in answer.replace(" ", ""):
            flags.append(f"context course code missing from answer: {code}")
    return flags


def abstained(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _ABSTAIN_PHRASES)


def score_qa(answer: str, chunks: list[dict[str, Any]], expected_behavior: str) -> dict[str, Any]:
    """Score one grounded-QA answer.

    ``expected_behavior``: 'answer' -> the context supports a real answer;
    'abstain' -> the context deliberately does NOT, and saying so is the correct output.
    """
    context = "\n".join(c.get("content", "") for c in chunks)
    faith = faithfulness_flags(answer, context)
    recall = recall_flags(answer, context)
    did_abstain = abstained(answer)
    return {
        "answer_chars": len(answer),
        "abstained": did_abstain,
        "behavior_ok": did_abstain if expected_behavior == "abstain" else not did_abstain,
        "faithfulness_flags": faith,
        "recall_flags": recall,
        "qa_auto_pass": (
            (did_abstain if expected_behavior == "abstain" else (not did_abstain and not faith))
        ),
        "needs_review": True,  # every free-text answer is manually graded, no exceptions
    }
