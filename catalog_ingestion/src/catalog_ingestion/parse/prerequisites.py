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


# ---------------------------------------------------------------------------
# Banner course-detail pages
# ---------------------------------------------------------------------------
#
# WHERE PREREQUISITES ACTUALLY LIVE. Neither of the two automated sources this pipeline uses
# publishes them, verified 2026-08-12:
#
#   * Acalog (catalog.purdue.edu, the requirements crawl) — a course detail page carries the
#     title, credits and description and nothing else. `parse_course_page` has always had a
#     `prerequisites_raw` field; it comes back None for every course.
#   * PurdueIO (api.purdue.io, the ~10.6k-course import) — the OData Course entity is
#     Id/Number/SubjectId/Title/CreditHours/Description. There is no prerequisite field at all.
#
# Purdue publishes them on Banner self-service (`bwckctlg.p_disp_course_detail`), whose
# robots.txt is `User-agent: * / Disallow: /` — a blanket disallow for automated clients. So
# there is no crawler here and there must not be one: this parses pages that a PERSON saved
# from their own browser, which is why the CLI that calls it (`import-prerequisites`) reads a
# directory instead of a URL.
#
# THE FORMAT, as Banner renders it. The detail body is one `<td class="ntdefault">` cell: free
# description text, then a run of labelled sections separated by `<br>`, each label in a
# `<span class="fieldlabeltext">` or bold. Course references inside them are `<a>` links.
#
#     <td class="ntdefault">
#       Discrete mathematics for computer science...
#       <br><br><span class="fieldlabeltext">Levels: </span>Undergraduate
#       <br><br><span class="fieldlabeltext">Prerequisites: </span>
#       Undergraduate level <a href="...">CS 18000</a> Minimum Grade of C
#       <br><br><span class="fieldlabeltext">Restrictions: </span>...
#     </td>
#
# Parsed by LABEL, not by position: sections vary per course and several are usually absent.
# Anything between a recognised label and the next one is that section's raw text; everything
# unrecognised is ignored rather than guessed at.
_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_NTDEFAULT_RE = re.compile(
    r'<td[^>]*class="[^"]*ntdefault[^"]*"[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL
)
# The course code Banner puts in the page title, e.g. "CS 18200 - Foundations Of Computer
# Science". `ddlabel` is the header cell class on the catalog detail page.
_TITLE_RE = re.compile(
    r'<t[hd][^>]*class="[^"]*ddlabel[^"]*"[^>]*>(?:\s*<a[^>]*>)?\s*'
    r'([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\s*-\s*([^<]*)',
    re.IGNORECASE,
)
# Every labelled section Banner emits that this parser knows how to file. The VALUES are the
# field names on `BannerCourseDetail`; a label not in here still terminates the previous
# section (so its text never bleeds into the one before it) but is not stored.
_SECTION_LABELS = {
    "prerequisites": "prerequisites_raw",
    "prerequisite": "prerequisites_raw",
    "corequisites": "corequisites_raw",
    "corequisite": "corequisites_raw",
    "restrictions": "restrictions_raw",
    "course attributes": "attributes_raw",
    "levels": None,
    "schedule types": None,
    "grading basis": None,
    "offered by": None,
    "department": None,
    "learning objectives": None,
    "may be offered at any of the following campuses": None,
    "repeatable for additional credit": None,
    "credit hours": None,
    "lecture hours": None,
    "lab hours": None,
    "other hours": None,
}
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /&']{2,60}?)\s*:\s*(.*)$")


@dataclass
class BannerCourseDetail:
    """One Banner course-detail page, reduced to the fields this pipeline stores."""

    course_code: str
    title: str
    prerequisites_raw: str | None = None
    corequisites_raw: str | None = None
    restrictions_raw: str | None = None
    attributes_raw: str | None = None
    description: str | None = None


def _visible_lines(html: str) -> list[str]:
    """Banner's `<br>`-separated body as plain lines, entities decoded, tags gone."""
    from html import unescape

    text = _BR_RE.sub("\n", html)
    text = _TAG_RE.sub(" ", text)
    lines = []
    for line in unescape(text).splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            lines.append(collapsed)
    return lines


def parse_banner_course_detail(html: str) -> BannerCourseDetail | None:
    """Parse one saved Banner `bwckctlg.p_disp_course_detail` page.

    Returns None when the page is not a course-detail page at all (a search form, an error, a
    login redirect) rather than guessing — a half-parsed page would write a course with no
    prerequisites, which is indistinguishable downstream from a course that genuinely has none.
    That distinction is the whole point of storing this.
    """
    title_match = _TITLE_RE.search(html)
    if not title_match:
        return None
    subject, number, title = title_match.groups()
    detail = BannerCourseDetail(
        course_code=f"{subject.upper()} {number.upper()}",
        title=" ".join(title.split()),
    )

    body_match = _NTDEFAULT_RE.search(html)
    if not body_match:
        return detail

    current: str | None = "description"
    buckets: dict[str, list[str]] = {}
    for line in _visible_lines(body_match.group(1)):
        label_match = _LABEL_RE.match(line)
        key = label_match.group(1).strip().lower() if label_match else None
        if key in _SECTION_LABELS:
            current = _SECTION_LABELS[key]
            remainder = label_match.group(2).strip()
            if current and remainder:
                buckets.setdefault(current, []).append(remainder)
            continue
        if current:
            buckets.setdefault(current, []).append(line)

    for field_name, lines in buckets.items():
        joined = " ".join(lines).strip()
        if joined:
            setattr(detail, field_name, joined)
    return detail
