"""Parse program overview metadata from preview_program.php pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from catalog_ingestion.parse.common import clean_text, soup

logger = logging.getLogger(__name__)

DEGREE_CODES = (
    "BS", "BA", "BFA", "BLA", "BM", "BSC", "BSE", "BSN", "BSPH",
    "MS", "MA", "MBA", "MFA", "PHD", "PHARMD", "DVM", "CERT",
)
PROGRAM_TYPE_MAP = {
    "minor": "minor",
    "certificate": "certificate",
    "concentration": "concentration",
    "honors": "honors",
    "tsap": "tsap",
    "transfer": "tsap",
    "graduate": "graduate",
    "doctoral": "graduate",
    "phd": "graduate",
}
SCHOOL_RE = re.compile(
    r"\b((?:College|School|Department|Division)\s+of\s+[A-Za-z ,&-]{3,60})", re.IGNORECASE
)
TOTAL_CREDITS_RE = re.compile(
    r"Total\s+(?:Minimum\s+)?(?:Credit\s+Hours?|Credits?)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Acalog renders the program total as a heading like "128 Credits Required".
CREDITS_REQUIRED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*Credits?\s*Required", re.IGNORECASE
)


@dataclass
class ParsedProgram:
    name: str
    degree_type: str | None
    program_type: str | None
    campus: str | None
    college_name: str | None
    total_credits_raw: str | None
    total_credits_min: float | None
    description: str | None
    poid: int | None
    source_url: str
    warnings: list[str] = field(default_factory=list)


def parse_program_page(html: str, url: str) -> ParsedProgram | None:
    """Parse a single program page. Returns None if no program found."""
    s = soup(html)
    for tag in s.find_all(["nav", "footer", "script", "style", "noscript"]):
        tag.decompose()

    poid: int | None = None
    qs = parse_qs(urlparse(url).query)
    poid_vals = qs.get("poid", [])
    if poid_vals:
        try:
            poid = int(poid_vals[0])
        except ValueError:
            pass

    title = _find_program_title(s, html)
    if not title:
        logger.warning("No program title found on %s", url)
        return None

    program_name, degree_type = _split_title(title)
    program_type = _infer_program_type(program_name, degree_type)
    college_name = _find_college(s)
    campus = _find_campus(s)
    description = _find_description(s)
    total_credits_raw, total_credits_min = _find_total_credits(s)

    warnings: list[str] = []
    if not degree_type:
        warnings.append("could not determine degree type from title")

    return ParsedProgram(
        name=program_name,
        degree_type=degree_type,
        program_type=program_type,
        campus=campus,
        college_name=college_name,
        total_credits_raw=total_credits_raw,
        total_credits_min=total_credits_min,
        description=description,
        poid=poid,
        source_url=url,
        warnings=warnings,
    )


def _find_program_title(s, html: str) -> str | None:
    h1 = s.find("h1")
    if h1:
        text = clean_text(h1.get_text())
        text = re.sub(r"\s*-\s*Purdue University.*$", "", text, flags=re.IGNORECASE)
        if text:
            return text

    title_tag = s.find("title")
    if title_tag:
        text = clean_text(title_tag.get_text())
        text = re.sub(r"\s*[|\-]\s*Purdue.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Program:\s*", "", text, flags=re.IGNORECASE)
        if text:
            return text

    for td in s.find_all("td"):
        text = clean_text(td.get_text())
        if re.search(r"\b(BS|BA|BFA|MS|MA|PHD|Minor|Certificate)\b", text, re.IGNORECASE):
            if len(text) < 200:
                return text

    return None


def _split_title(title: str) -> tuple[str, str | None]:
    """Split 'Computer Science: Machine Intelligence, BS' → ('Computer Science: Machine Intelligence', 'BS')."""
    parts = [p.strip() for p in title.rsplit(",", 1)]
    if len(parts) == 2:
        potential_degree = parts[1].upper().replace(".", "").strip()
        for code in DEGREE_CODES:
            if potential_degree.startswith(code):
                return parts[0], potential_degree
    return title, None


def _infer_program_type(name: str, degree: str | None) -> str | None:
    combined = f"{name} {degree or ''}".lower()
    for keyword, ptype in PROGRAM_TYPE_MAP.items():
        if keyword in combined:
            return ptype
    if degree and degree.startswith(("BS", "BA", "BFA", "BLA", "BM")):
        return "undergraduate_major"
    if degree and degree.startswith(("MS", "MA", "MBA", "PHD")):
        return "graduate"
    return "major"


def _find_college(s) -> str | None:
    text = s.get_text()
    match = SCHOOL_RE.search(text[:3000])
    if match:
        return clean_text(match.group(1)).rstrip(" ,")
    return None


def _find_campus(s) -> str | None:
    text = s.get_text()
    campus_re = re.compile(r"\b(West Lafayette|Indianapolis|Fort Wayne|Northwest)\b", re.IGNORECASE)
    match = campus_re.search(text[:2000])
    return match.group(1) if match else None


def _find_description(s) -> str | None:
    # Acalog: the "About the Program" acalog-core block holds the overview text.
    for core in s.select("div.acalog-core"):
        heading = core.find(["h1", "h2", "h3", "h4"])
        if heading and "about the program" in clean_text(heading.get_text()).lower():
            heading.extract()
            text = clean_text(core.get_text(" "))
            if text:
                return text
    for selector in ["div.block_content p", "div.program_description", "div#program-description"]:
        el = s.select_one(selector)
        if el:
            text = clean_text(el.get_text())
            if text:
                return text
    return None


def _find_total_credits(s) -> tuple[str | None, float | None]:
    text = clean_text(s.get_text(" "))
    for pattern in (TOTAL_CREDITS_RE, CREDITS_REQUIRED_RE):
        match = pattern.search(text)
        if match:
            try:
                return match.group(0), float(match.group(1))
            except ValueError:
                return match.group(0), None
    return None, None
