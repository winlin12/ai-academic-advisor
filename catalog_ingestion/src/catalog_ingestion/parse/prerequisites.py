"""Parse prerequisite text into structured JSON.

The raw prerequisite text is ALWAYS preserved. The parsed JSON is best-effort.
When parsing is ambiguous, confidence is set to 'low' and raw_text is kept.

Example input:
  "Prerequisite: CS 18000 and (MA 16100 or MA 16500)."

Example output:
  {
    "type": "AND",
    "children": [
      {"type": "COURSE", "course": "CS 18000"},
      {
        "type": "OR",
        "children": [
          {"type": "COURSE", "course": "MA 16100"},
          {"type": "COURSE", "course": "MA 16500"}
        ]
      }
    ]
  }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

PREREQ_PREFIX_RE = re.compile(r"^\s*(?:Prerequisite|Prerequisites)\s*[:\-]?\s*", re.IGNORECASE)
COURSE_RE = re.compile(r"\b([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\b")
GRADE_RE = re.compile(r"(?:grade of|minimum grade)\s+([A-D][+-]?|P)\b", re.IGNORECASE)
CONCURRENT_RE = re.compile(r"\b(concurrently|concurrent enrollment)\b", re.IGNORECASE)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


@dataclass
class ParseResult:
    raw_text: str
    parsed_json: dict[str, Any] | None
    parse_confidence: str
    parser_notes: str


def parse_prerequisite_text(raw: str) -> ParseResult:
    """Attempt to parse prerequisite text. Always preserves raw_text."""
    if not raw or not raw.strip():
        return ParseResult(raw_text=raw, parsed_json=None,
                           parse_confidence=CONFIDENCE_HIGH, parser_notes="empty")

    cleaned = PREREQ_PREFIX_RE.sub("", raw).strip()
    cleaned = cleaned.rstrip(".")

    if not cleaned:
        return ParseResult(raw_text=raw, parsed_json=None,
                           parse_confidence=CONFIDENCE_HIGH, parser_notes="prefix-only")

    course_matches = list(COURSE_RE.finditer(cleaned.upper()))
    if not course_matches:
        return ParseResult(
            raw_text=raw, parsed_json=None,
            parse_confidence=CONFIDENCE_LOW,
            parser_notes="no course references found",
        )

    try:
        parsed, notes = _parse_expr(cleaned.upper())
        confidence = CONFIDENCE_HIGH if not notes else CONFIDENCE_MEDIUM
        return ParseResult(raw_text=raw, parsed_json=parsed,
                           parse_confidence=confidence, parser_notes="; ".join(notes))
    except Exception as exc:
        logger.debug("Prerequisite parse failed for %r: %s", raw[:80], exc)
        return ParseResult(
            raw_text=raw, parsed_json=None,
            parse_confidence=CONFIDENCE_LOW,
            parser_notes=f"parse error: {exc}",
        )


def _parse_expr(text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a prerequisite expression into an AND/OR tree."""
    notes: list[str] = []

    # Split on top-level ' AND ' first (highest binding)
    and_parts = _split_top_level(text, " AND ")
    if len(and_parts) > 1:
        children = []
        for part in and_parts:
            child, child_notes = _parse_expr(part.strip())
            children.append(child)
            notes.extend(child_notes)
        return {"type": "AND", "children": children}, notes

    or_parts = _split_top_level(text, " OR ")
    if len(or_parts) > 1:
        children = []
        for part in or_parts:
            child, child_notes = _parse_expr(part.strip())
            children.append(child)
            notes.extend(child_notes)
        return {"type": "OR", "children": children}, notes

    # Strip outer parens
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        return _parse_expr(inner)

    # Single course reference
    m = COURSE_RE.search(stripped)
    if m:
        course_code = f"{m.group(1)} {m.group(2)}"
        node: dict[str, Any] = {"type": "COURSE", "course": course_code}
        grade_m = GRADE_RE.search(stripped)
        if grade_m:
            node["minimum_grade"] = grade_m.group(1)
        if CONCURRENT_RE.search(stripped):
            node["concurrent_allowed"] = True
        return node, notes

    notes.append(f"unrecognized expression: {stripped[:60]!r}")
    return {"type": "UNKNOWN", "raw": stripped}, notes


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split text on separator only when not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    sep_len = len(separator)
    i = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
            current.append(text[i])
            i += 1
        elif text[i] == ")":
            depth -= 1
            current.append(text[i])
            i += 1
        elif depth == 0 and text[i:i + sep_len] == separator:
            parts.append("".join(current))
            current = []
            i += sep_len
        else:
            current.append(text[i])
            i += 1
    parts.append("".join(current))
    return parts if len(parts) > 1 else parts
