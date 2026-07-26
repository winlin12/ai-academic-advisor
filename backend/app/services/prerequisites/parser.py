"""Parse Purdue Banner prerequisite HTML into a structured AND/OR tree and a flat group form.

WHERE THIS DATA COMES FROM, AND THE CATCH. Purdue publishes prerequisites in exactly one
machine-readable place: the Banner Schedule-of-Classes course-detail page
(``selfservice.mypurdue.purdue.edu/prod/bwckctlg.p_disp_course_detail``). Neither the Acalog
catalog nor the PurdueIO API carries them (TODO.md §2.5). That host's ``robots.txt`` is a
blanket ``Disallow: /`` — so the crawler that feeds this parser (``sync.py``) is run
deliberately, at low volume, with a truthful User-Agent and a disk cache, as an explicit
project decision, NOT as an unattended background crawl. This module is pure parsing and
touches no network; that separation is on purpose.

TWO OUTPUT SHAPES, because two consumers need different fidelity:

  parse_tree()   -> the full AND/OR/COURSE tree, preserving minimum grades and
                    "may be taken concurrently". Stored as ``parsed_json`` so nothing the
                    source stated is thrown away.
  to_groups()    -> a flat AND-of-OR-groups: ``[["CS 25000"], ["CS 25100", "CS 25300", ...]]``
                    means "CS 25000 AND (one of CS 25100/CS 25300/...)". This is what the
                    deterministic planner can actually evaluate. It is a LOSSY reduction and
                    returns None when the tree is too deep to flatten honestly (a nested AND
                    inside an OR), so a caller never mistakes "couldn't reduce" for "no prereq".

The Banner text is noisy — every option reads "Undergraduate level CS 25100 Minimum Grade of C
[may be taken concurrently]". The structural signal is only the course codes, the and/or
words, and the parentheses; everything else is stripped before the tree parse.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

# Subject + number, tolerant of the trailing letter some courses carry (e.g. MA 16021, CS 490H).
_COURSE_RE = re.compile(r"\b([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\b")
_GRADE_RE = re.compile(r"Minimum Grade of\s+([A-DP][+-]?)", re.IGNORECASE)
# The cleaner rewrites Banner's "[may be taken concurrently]" to this inline sentinel and
# leaves it ADJACENT to its course, so per-course concurrency survives the split into leaves.
# Stripping the phrase outright (the old behaviour) lost it — and a co-requisite wrongly
# modelled as a hard prerequisite forces an ordering the real rule does not, which is exactly
# the failure that made CS 18000 look like it required calculus a term early.
_CONCURRENT_TOKEN = "~CONCURRENT~"
_CONCURRENT_RE = re.compile(re.escape(_CONCURRENT_TOKEN))

# Footer boilerplate that follows the prerequisite block on a Banner detail page. The block
# has no closing tag of its own, so these sentinels are how its END is found; without them the
# page footer ("Return to Previous", the ITaP help line) leaks into the last OR option and
# quietly drops the parser to medium confidence.
_FOOTER_SENTINELS = (
    "Return to Previous", "Skip to top of page", "Having trouble?",
    "Corequisites:", "Restrictions:", "Course Attributes:",
)


@dataclass
class PrereqParse:
    raw_text: str                       # the human-readable expression, always preserved
    tree: dict[str, Any] | None         # AND/OR/COURSE tree, or None if nothing parseable
    groups: list[list[str]] | None      # flat AND-of-ORs for the planner, or None if unflattenable
    confidence: str                     # 'high' | 'medium' | 'low' | 'none'
    notes: list[str] = field(default_factory=list)

    @property
    def all_codes(self) -> list[str]:
        """Every course code mentioned anywhere, deduped, for foreign-key resolution."""
        seen, out = set(), []
        for code in _COURSE_RE.findall(" ".join(_flatten_codes(self.tree))):
            joined = f"{code[0]} {code[1]}"
            if joined not in seen:
                seen.add(joined)
                out.append(joined)
        return out


def extract_prereq_block(page_html: str) -> str | None:
    """Slice the prerequisite region out of a Banner course-detail page.

    Returns None when the page has no prerequisite block at all — which is itself information
    (the course has no prerequisites), distinct from a fetch that failed.
    """
    start = page_html.find("Prerequisites:")
    if start < 0:
        return None
    tail = page_html[start:]
    # End at the nearest footer sentinel or the enclosing cell, whichever comes first.
    end = len(tail)
    for sentinel in (*_FOOTER_SENTINELS, "</td>"):
        stripped = re.sub(r"<[^>]+>", "", tail)
        idx = stripped.find(sentinel)
        # search in the tag-stripped text but cut the raw tail proportionally is fragile;
        # instead cut the raw tail at the sentinel found in the raw text when present.
        raw_idx = tail.find(sentinel)
        if raw_idx > 0:
            end = min(end, raw_idx)
    return tail[:end]


def _clean_to_expression(block_html: str) -> str:
    """Banner prereq HTML -> 'CS 25000 and (CS 25100 or CS 25300)'-style expression."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", block_html))
    text = text.replace("\xa0", " ")
    text = text.replace("Prerequisites:", " ")
    # Drop the level qualifier, the grade requirement, the concurrency note, and the newer
    # "Requisites / General Requirements / Course or Test:" table scaffolding. None of these
    # change WHICH courses gate the course; they are recorded separately by parse_tree().
    text = re.sub(r"\b(?:Undergraduate|Graduate|Professional) level\b", " ", text, flags=re.I)
    text = re.sub(r"Minimum Grade of\s+[A-DP][+-]?", " ", text, flags=re.I)
    # "may NOT be taken concurrently" is just a normal (strict) prereq — drop it. "may be taken
    # concurrently" is a co-requisite — keep it, inline, so the leaf it belongs to is flagged.
    text = re.sub(r"\[?may not be taken concurrently\]?\.?", " ", text, flags=re.I)
    text = re.sub(r"\[?may be taken concurrently\]?\.?", f" {_CONCURRENT_TOKEN} ", text, flags=re.I)
    text = re.sub(r"\b[A-Z]{2,6}\s+\d{3,5}[A-Z]?\s+Requisites\b", " ", text)
    text = re.sub(r"General Requirements:", " ", text, flags=re.I)
    text = re.sub(r"Course or Test:", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate at the last course code (plus any trailing close-parens): a Banner detail page
    # has no closing tag on the prereq block, so despite the footer sentinels some page footer
    # ("Return to Previous", onMouseOver scripts) can still trail the expression. Everything
    # after the final course token is by definition not part of the boolean expression.
    last = None
    for last in _COURSE_RE.finditer(text.upper()):
        pass
    if last:
        cut = last.end()
        # Keep trailing close-parens and a concurrency sentinel that belongs to the last course;
        # everything past that is page footer, not part of the boolean expression.
        tail = text[cut:]
        keep = re.match(rf"^(?:\s|\)|{re.escape(_CONCURRENT_TOKEN)})*", tail)
        text = text[:cut + (keep.end() if keep else 0)].strip()
    return text


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on ``separator`` only outside parentheses (so 'A and (B or C)' splits on 'and')."""
    parts, depth, current, i = [], 0, [], 0
    sep = separator
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
            current.append(ch); i += 1
        elif ch == ")":
            depth -= 1
            current.append(ch); i += 1
        elif depth == 0 and text[i:i + len(sep)] == sep:
            parts.append("".join(current)); current = []; i += len(sep)
        else:
            current.append(ch); i += 1
    parts.append("".join(current))
    return parts


def _parse_expr(text: str, notes: list[str]) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # AND binds looser than OR in Banner's fully-parenthesised expressions: split AND first so
    # each side becomes its own subtree, then OR within.
    for op, kind in ((" and ", "AND"), (" or ", "OR")):
        parts = _split_top_level(text, op)
        if len(parts) > 1:
            children = [c for c in (_parse_expr(p, notes) for p in parts) if c]
            if not children:
                return None
            return children[0] if len(children) == 1 else {"type": kind, "children": children}
    if text.startswith("(") and text.endswith(")"):
        return _parse_expr(text[1:-1], notes)
    match = _COURSE_RE.search(text.upper())
    if match:
        node: dict[str, Any] = {"type": "COURSE", "course": f"{match.group(1)} {match.group(2)}"}
        grade = _GRADE_RE.search(text)
        if grade:
            node["minimum_grade"] = grade.group(1).upper()
        if _CONCURRENT_RE.search(text):
            node["concurrent_allowed"] = True
        return node
    notes.append(f"unrecognized fragment: {text[:50]!r}")
    return None


def to_groups(tree: dict[str, Any] | None) -> list[list[str]] | None:
    """Flatten an AND/OR tree to AND-of-OR-groups, or None if it can't be flattened honestly.

    A nested AND inside an OR ("(A and B) or C") has no faithful flat representation, so this
    returns None rather than silently approximating — the caller keeps the full tree and knows
    the flat planner cannot enforce this particular rule.
    """
    if tree is None:
        return []
    kind = tree.get("type")
    if kind == "COURSE":
        return [[tree["course"]]]
    if kind == "OR":
        codes: list[str] = []
        for child in tree.get("children", []):
            if child.get("type") != "COURSE":
                return None  # OR over a compound -> not flattenable
            if child["course"] not in codes:  # Banner repeats options; dedupe, keep order
                codes.append(child["course"])
        return [codes]
    if kind == "AND":
        groups: list[list[str]] = []
        for child in tree.get("children", []):
            sub = to_groups(child)
            if sub is None:
                return None
            groups.extend(sub)
        return groups
    return None


def _flatten_codes(tree: dict[str, Any] | None) -> list[str]:
    if not tree:
        return []
    if tree.get("type") == "COURSE":
        return [tree["course"]]
    out: list[str] = []
    for child in tree.get("children", []):
        out.extend(_flatten_codes(child))
    return out


def parse_prereq_html(page_html: str) -> PrereqParse:
    """Full pipeline: Banner page HTML -> PrereqParse. The single entry point for callers."""
    block = extract_prereq_block(page_html)
    if block is None:
        return PrereqParse(raw_text="", tree=None, groups=[], confidence="none",
                           notes=["no prerequisite block on page"])
    return parse_prereq_expression(_clean_to_expression(block))


def parse_prereq_expression(expression: str) -> PrereqParse:
    """Parse an already-cleaned 'A and (B or C)' expression. Split out so it is unit-testable
    without an HTML fixture."""
    if not expression or not _COURSE_RE.search(expression.upper()):
        return PrereqParse(raw_text=expression, tree=None, groups=[], confidence="none",
                           notes=["no course codes"])
    notes: list[str] = []
    tree = _parse_expr(expression, notes)
    groups = to_groups(tree)
    if tree is None:
        confidence = "low"
    elif notes or groups is None:
        confidence = "medium"
    else:
        confidence = "high"
    return PrereqParse(raw_text=expression, tree=tree, groups=groups,
                       confidence=confidence, notes=notes)
