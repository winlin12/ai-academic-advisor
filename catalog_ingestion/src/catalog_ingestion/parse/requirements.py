"""Parse degree requirement sections from Acalog program pages.

Acalog renders a program's requirements, plan of study, and policy notes as a flat
series of ``div.acalog-core`` blocks. Each block has a heading (h2/h3/h4/h5 — the tag
level encodes the logical nesting) and may contain a ``<ul>`` of courses, each rendered
as ``<li class="acalog-course">`` with an ``<a onclick="showCourse('<catoid>','<coid>',...)">``
link plus ``Credit Hours:`` markup.

This parser:
1. Walks every ``div.acalog-core`` in document order and rebuilds the heading hierarchy.
2. Extracts each course option with its coid, course code, title, and credit hours.
3. Captures the credit total from headings like "(12 credits)" / "78-79 credits".
4. Always preserves the block's raw text — narrative/policy blocks with no courses are
   kept as groups so requirement text is never silently discarded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from catalog_ingestion.parse.common import clean_text, soup

logger = logging.getLogger(__name__)

# showCourse('<catoid>', '<coid>', this, '<serialized>') — coid is the 2nd argument.
SHOW_COURSE_RE = re.compile(r"showCourse\(\s*'(\d+)'\s*,\s*'(\d+)'")
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\b")
# Credits inside a heading, e.g. "(12 credits)", "(78-79 credits)", "15-16 Credits",
# "128 Credits Required".
HEADING_CREDITS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|to)?\s*(\d+(?:\.\d+)?)?\s*credits?",
    re.IGNORECASE,
)
LI_CREDITS_RE = re.compile(
    r"Credit\s+Hours?:\s*(\d+(?:\.\d+)?)(?:\s*(?:or|to|-|–)\s*(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
MIN_GRADE_RE = re.compile(r"Minimum\s+Grade(?:\s+of)?\s*([A-D][+-]?)", re.IGNORECASE)
SELECTIVE_MARKERS = re.compile(
    r"\bchoose\b|\bselect\b|\bone of\b|\bany of\b|\belective", re.IGNORECASE
)

# Heading keyword → requirement_type. Order matters (first match wins).
REQUIREMENT_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("sample 4-year plan", "plan_of_study"),
    ("plan of study", "plan_of_study"),
    ("4-year plan", "plan_of_study"),
    ("about the program", "overview"),
    ("learning outcome", "learning_outcomes"),
    ("disclaimer", "disclaimer"),
    ("pre-requisite", "prerequisite_info"),
    ("prerequisite", "prerequisite_info"),
    ("critical course", "critical_course"),
    ("world language", "world_language"),
    ("gpa requirement", "gpa"),
    ("licensure", "licensure"),
    ("transfer credit", "policy"),
    ("pass/no pass", "policy"),
    ("pass / no pass", "policy"),
    ("university core", "core"),
    ("core curriculum", "core"),
    ("civics literacy", "university"),
    ("upper level requirement", "university"),
    ("university requirement", "university"),
    ("college", "college"),
    ("foundation", "foundation"),
    ("concentration", "concentration"),
    ("free elective", "free_elective"),
    ("elective", "elective"),
    ("selective", "selective"),
    ("professional education", "major"),
    ("major course", "major"),
    ("major requirement", "major"),
    ("departmental", "major"),
    ("degree requirement", "degree"),
    ("requirement", "requirement"),
)

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass
class RequirementOptionData:
    course_code_raw: str | None
    option_text: str
    credits: float | None
    is_selective: bool
    display_order: int
    coid: int | None = None
    title: str | None = None
    credits_max: float | None = None
    minimum_grade: str | None = None


@dataclass
class RequirementGroupData:
    name: str
    requirement_type: str | None
    credits_min: float | None
    credits_max: float | None
    raw_text: str
    display_order: int
    options: list[RequirementOptionData] = field(default_factory=list)
    children: list["RequirementGroupData"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_requirement_sections(html: str) -> list[RequirementGroupData]:
    """Parse all requirement sections from a program page into a hierarchy.

    Returns the top-level requirement groups; nested groups are attached via
    ``RequirementGroupData.children``. The "About the Program" overview block is
    included (type ``overview``) — callers may use it for the program description.
    """
    s = soup(html)
    for tag in s.find_all(["script", "style", "noscript"]):
        tag.decompose()

    cores = s.select("div.acalog-core")
    if not cores:
        logger.warning("No acalog-core blocks found in program page")
        return []

    roots: list[RequirementGroupData] = []
    # stack of (heading_level, group); shallower level number = higher in hierarchy.
    stack: list[tuple[int, RequirementGroupData]] = []
    order = 0

    for core in cores:
        order += 1
        group, level = _parse_core_block(core, order)

        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            parent.children.append(group)
            # A sub-section with no type of its own belongs to its parent's category
            # (e.g. "Fall 1st Year" under a "Sample 4-Year Plan" plan_of_study block).
            if group.requirement_type is None:
                group.requirement_type = parent.requirement_type
        else:
            roots.append(group)
        stack.append((level, group))

    total = _count_groups(roots)
    logger.info("Parsed %d requirement groups (%d top-level)", total, len(roots))
    return roots


def _count_groups(groups: list[RequirementGroupData]) -> int:
    return sum(1 + _count_groups(g.children) for g in groups)


def _parse_core_block(core, display_order: int) -> tuple[RequirementGroupData, int]:
    heading_el = _first_heading(core)
    if heading_el is not None:
        name = clean_text(heading_el.get_text())
        level = int(heading_el.name[1])
    else:
        name = ""
        level = 9  # no heading → treat as a deep child of the preceding block

    credits_min, credits_max = _credits_from_heading(name)
    req_type = _infer_requirement_type(name)
    raw_text = _direct_text(core)
    selective_group = bool(SELECTIVE_MARKERS.search(name))

    options = _parse_course_options(core, selective_group)

    return (
        RequirementGroupData(
            name=name or "(untitled section)",
            requirement_type=req_type,
            credits_min=credits_min,
            credits_max=credits_max,
            raw_text=raw_text,
            display_order=display_order,
            options=options,
        ),
        level,
    )


def _first_heading(core):
    for child in core.descendants:
        if getattr(child, "name", None) in _HEADING_TAGS:
            # Only count a heading that belongs to this block, not a nested acalog-core.
            ancestor = child.find_parent("div", class_="acalog-core")
            if ancestor is core:
                return child
    return None


def _direct_text(core) -> str:
    """Text of this block excluding any nested acalog-core descendants."""
    clone = BeautifulSoup(str(core), "lxml")
    root = clone.find("div", class_="acalog-core")
    if root is None:
        return clean_text(core.get_text(" "))
    for nested in root.select("div.acalog-core"):
        if nested is not root:
            nested.decompose()
    # Preserve line structure for readability, then collapse within lines.
    text = root.get_text("\n")
    lines = [clean_text(ln) for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _parse_course_options(core, selective_group: bool) -> list[RequirementOptionData]:
    options: list[RequirementOptionData] = []
    order = 0
    for li in core.find_all("li", class_="acalog-course"):
        # Skip course items that live in a nested acalog-core (handled by that block).
        if li.find_parent("div", class_="acalog-core") is not core:
            continue
        order += 1
        options.append(_parse_course_li(li, order, selective_group))
    return options


def _parse_course_li(li, display_order: int, selective_group: bool) -> RequirementOptionData:
    full_text = clean_text(li.get_text(" "))

    coid: int | None = None
    a = li.find("a")
    if a is not None:
        onclick = a.get("onclick", "") or ""
        m = SHOW_COURSE_RE.search(onclick)
        if m:
            coid = int(m.group(2))
        link_text = clean_text(a.get_text())
    else:
        link_text = full_text

    course_code: str | None = None
    title: str | None = None
    code_match = COURSE_CODE_RE.search(link_text)
    if code_match:
        course_code = f"{code_match.group(1)} {code_match.group(2)}"
        remainder = link_text[code_match.end():].lstrip(" -–—: ").strip()
        title = remainder or None

    credits: float | None = None
    credits_max: float | None = None
    cm = LI_CREDITS_RE.search(full_text)
    if cm:
        credits = _to_float(cm.group(1))
        credits_max = _to_float(cm.group(2)) if cm.group(2) else credits

    grade_match = MIN_GRADE_RE.search(full_text)
    minimum_grade = grade_match.group(1).upper() if grade_match else None

    is_selective = selective_group or bool(re.search(r"\bor\b", full_text))

    return RequirementOptionData(
        course_code_raw=course_code,
        option_text=full_text,
        credits=credits,
        credits_max=credits_max,
        is_selective=is_selective,
        display_order=display_order,
        coid=coid,
        title=title,
        minimum_grade=minimum_grade,
    )


def _credits_from_heading(heading: str) -> tuple[float | None, float | None]:
    if not heading or "credit" not in heading.lower():
        return None, None
    m = HEADING_CREDITS_RE.search(heading)
    if not m:
        return None, None
    lo = _to_float(m.group(1))
    hi = _to_float(m.group(2)) if m.group(2) else lo
    return lo, hi


def _infer_requirement_type(heading: str) -> str | None:
    low = heading.lower()
    for keyword, req_type in REQUIREMENT_TYPE_RULES:
        if keyword in low:
            return req_type
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
