"""Parse individual course pages from the Purdue catalog.

Course pages follow the URL pattern:
  /preview_course_nopop.php?catoid=N&coid=XXXXX

The page contains course code, title, credit hours, description,
prerequisites, corequisites, restrictions, and attributes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from catalog_ingestion.parse.common import clean_text, soup

logger = logging.getLogger(__name__)

COURSE_CODE_HEADER_RE = re.compile(r"([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\s*[-–]?\s*(.*)", re.DOTALL)
# Acalog renders the credit hours inline: "Credit Hours: 3.00." or "Credit Hours: 1.00 to 18.00."
CREDIT_HOURS_RE = re.compile(
    r"Credit\s+Hours?:\s*(\d+(?:\.\d+)?)(?:\s*(?:to|-|–)\s*(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
# Section markers that terminate the description and label structured fields. Order is
# longest-first so e.g. "Prerequisite/Corequisite" matches before "Prerequisite".
SECTION_MARKERS = (
    "Prerequisite/Corequisite",
    "Prerequisites/Corequisites",
    "Prerequisites",
    "Prerequisite",
    "Corequisites",
    "Corequisite",
    "Registration Restrictions",
    "Restrictions",
    "Restriction",
    "Learning Outcomes",
    "Attributes",
    "Repeatable",
    "Typically Offered",
    "Schedule Types",
    "Course Attributes",
)
_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in SECTION_MARKERS) + r")\b\s*:?\s*",
    re.IGNORECASE,
)


@dataclass
class ParsedCourse:
    subject_code: str
    course_number: str
    course_code: str
    coid: int | None
    title: str | None
    description: str | None
    credit_hours_min: float | None
    credit_hours_max: float | None
    credit_hours_raw: str | None
    prerequisites_raw: str | None
    corequisites_raw: str | None
    restrictions_raw: str | None
    attributes_raw: str | None
    source_url: str
    warnings: list[str] = field(default_factory=list)


def parse_course_page(html: str, url: str) -> ParsedCourse | None:
    """Parse a single Acalog course detail page (preview_course[_nopop].php).

    The content lives in ``td.block_content`` with ``<h1 id="course_preview_title">`` as
    the course code+title, followed by inline "Credit Hours: X.XX." then the description,
    with structured fields (Prerequisites, Restrictions, ...) and Learning Outcomes after.
    """
    s = soup(html)
    for tag in s.find_all(["script", "style", "noscript"]):
        tag.decompose()

    coid: int | None = None
    qs = parse_qs(urlparse(url).query)
    coid_vals = qs.get("coid", [])
    if coid_vals:
        try:
            coid = int(coid_vals[0])
        except ValueError:
            pass

    heading_el = _find_course_heading_el(s)
    if heading_el is None:
        logger.warning("No course heading found on %s", url)
        return None

    title_text = clean_text(heading_el.get_text())
    header_match = COURSE_CODE_HEADER_RE.match(title_text)
    if not header_match:
        logger.warning("Could not parse course code from heading %r on %s", title_text[:80], url)
        return None

    subject_code = header_match.group(1).strip()
    course_number = header_match.group(2).strip()
    course_code = f"{subject_code} {course_number}"
    title = clean_text(header_match.group(3)) or None

    # Everything after the heading, within the same content block, as clean text.
    body_text = _body_text_after_heading(heading_el)

    cr_min = cr_max = None
    cr_raw: str | None = None
    cr_match = CREDIT_HOURS_RE.search(body_text)
    if cr_match:
        cr_raw = cr_match.group(0)
        cr_min = float(cr_match.group(1))
        cr_max = float(cr_match.group(2)) if cr_match.group(2) else cr_min

    description = _extract_description(body_text, cr_match)
    prereqs_raw = _extract_field(body_text, ("Prerequisite/Corequisite",
                                             "Prerequisites/Corequisites",
                                             "Prerequisites", "Prerequisite"))
    coreqs_raw = _extract_field(body_text, ("Corequisites", "Corequisite"))
    restrictions_raw = _extract_field(body_text, ("Registration Restrictions",
                                                  "Restrictions", "Restriction"))
    attrs_raw = _extract_field(body_text, ("Course Attributes", "Attributes"))

    return ParsedCourse(
        subject_code=subject_code,
        course_number=course_number,
        course_code=course_code,
        coid=coid,
        title=title,
        description=description,
        credit_hours_min=cr_min,
        credit_hours_max=cr_max,
        credit_hours_raw=cr_raw,
        prerequisites_raw=prereqs_raw,
        corequisites_raw=coreqs_raw,
        restrictions_raw=restrictions_raw,
        attributes_raw=attrs_raw,
        source_url=url,
        warnings=[],
    )


def _find_course_heading_el(s):
    """Locate the course title element (prefer Acalog's id, then any course-code heading)."""
    el = s.find(id="course_preview_title")
    if el is not None and re.match(r"[A-Z]{2,6}\s+\d{3,5}", clean_text(el.get_text())):
        return el
    for h in s.find_all(["h1", "h2", "h3"]):
        if re.match(r"[A-Z]{2,6}\s+\d{3,5}", clean_text(h.get_text())):
            return h
    return None


def _body_text_after_heading(heading_el) -> str:
    """Clean text of the content following the heading, bounded to its block."""
    parts = [str(sib) for sib in heading_el.next_siblings]
    fragment = "".join(parts)
    if not fragment.strip():
        # Heading wrapped separately — fall back to the enclosing content block.
        container = heading_el.find_parent("td", class_="block_content") or heading_el.parent
        if container is not None:
            text = container.get_text("\n")
            # Drop everything up to and including the heading text itself.
            idx = text.find(heading_el.get_text())
            if idx != -1:
                text = text[idx + len(heading_el.get_text()):]
            return _clean_block(text)
    return _clean_block(soup(fragment).get_text("\n"))


def _clean_block(text: str) -> str:
    lines = [clean_text(ln) for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_description(body_text: str, cr_match) -> str | None:
    """Description = text after the credit-hours sentence up to the first section marker."""
    start = cr_match.end() if cr_match else 0
    rest = body_text[start:].lstrip(" .\n")
    marker = _MARKER_RE.search(rest)
    if marker:
        rest = rest[: marker.start()]
    desc = clean_text(rest)
    return desc or None


def _extract_field(body_text: str, labels: tuple[str, ...]) -> str | None:
    """Return the text following a label up to the next section marker. Never guesses."""
    for label in labels:
        m = re.search(r"\b" + re.escape(label) + r"\b\s*:?\s*", body_text, re.IGNORECASE)
        if not m:
            continue
        rest = body_text[m.end():]
        nxt = _MARKER_RE.search(rest)
        if nxt:
            rest = rest[: nxt.start()]
        value = clean_text(rest)
        if value:
            return value
    return None


def extract_course_links_from_listing(html: str, base_url: str, catoid: int) -> list[tuple[str, int]]:
    """Extract (url, coid) pairs from a course listing/filter page.

    Acalog also links courses via preview_course.php (popup) in some templates, so we
    accept both preview_course_nopop.php and preview_course.php. Hrefs are HTML-escaped
    (?catoid=19&amp;coid=...), so we unescape before building the URL — otherwise the
    fetched URL carries a literal `&amp;` and the server never sees the `coid` param.
    """
    import html as _html
    from urllib.parse import urljoin
    coid_re = re.compile(
        r'href=["\']([^"\']*preview_course(?:_nopop)?\.php[^"\']*coid=(\d+)[^"\']*)["\']',
        re.IGNORECASE,
    )
    results: list[tuple[str, int]] = []
    seen: set[int] = set()
    for match in coid_re.finditer(html):
        href = _html.unescape(match.group(1))
        coid = int(match.group(2))
        if coid in seen:
            continue
        seen.add(coid)
        full_url = urljoin(base_url, href)
        results.append((full_url, coid))
    return results
