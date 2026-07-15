"""Discover subject codes and build course-listing URLs from the Acalog courses filter.

The courses page (content.php?catoid=N&navoid=M) renders a filter form whose subject
dropdown is ``<select id="courseprefix" name="filter[<k>]">``. Selecting a prefix and
submitting reloads the page with the matching courses (100 per page, paginated via
``filter[cpage]``). Each course is an ``<a href="preview_course_nopop.php?...coid=...">``.

The numeric filter index ``<k>`` (27 for catoid 19) is catalog-specific, so we discover
it from the dropdown's ``name`` attribute rather than hard-coding it everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlencode, urljoin

from catalog_ingestion.parse.common import clean_text, soup

logger = logging.getLogger(__name__)

# navoid for the "Courses" filter page per known catoid.
KNOWN_COURSES_NAVOIDS: dict[int, int] = {
    19: 25468,  # 2026-2027
}

# Default subject-prefix filter index for the current catalog. Discovered dynamically
# from the courses page when possible (see discover_prefix_filter_param).
DEFAULT_PREFIX_PARAM = "filter[27]"


@dataclass
class SubjectInfo:
    code: str
    name: str | None


def build_courses_page_url(base_url: str, catoid: int, navoid: int) -> str:
    """URL of the unfiltered courses filter page (used to read the subject dropdown)."""
    return urljoin(base_url, f"/content.php?catoid={catoid}&navoid={navoid}")


def discover_prefix_filter_param(html: str) -> str:
    """Return the subject-prefix filter param name, e.g. 'filter[27]'.

    Falls back to DEFAULT_PREFIX_PARAM if the dropdown can't be located.
    """
    s = soup(html)
    sel = s.find("select", id="courseprefix")
    if sel and sel.get("name"):
        return sel["name"]
    # Fall back: any select whose options look like subject codes.
    for sel in s.find_all("select"):
        name = sel.get("name", "")
        if name.startswith("filter[") and any(
            (o.get("value") or "").isalpha() for o in sel.find_all("option")
        ):
            return name
    logger.warning("Could not find course-prefix select; using default %s", DEFAULT_PREFIX_PARAM)
    return DEFAULT_PREFIX_PARAM


def discover_subjects_from_filter_page(html: str) -> list[SubjectInfo]:
    """Parse subject codes from the course-prefix dropdown."""
    s = soup(html)
    sel = s.find("select", id="courseprefix")
    options = sel.find_all("option") if sel else []
    if not options:
        # Fall back to any filter select with alpha option values.
        for cand in s.find_all("select"):
            if cand.get("name", "").startswith("filter[") and cand.find_all("option"):
                options = cand.find_all("option")
                break

    subjects: list[SubjectInfo] = []
    seen: set[str] = set()
    for opt in options:
        code = (opt.get("value") or "").strip().upper()
        if not code or code == "-1" or not code.isalpha():
            continue
        if code in seen:
            continue
        seen.add(code)
        label = clean_text(opt.get_text())
        name = label if label and label.upper() != code else None
        subjects.append(SubjectInfo(code=code, name=name))

    subjects.sort(key=lambda s: s.code)
    logger.info("Discovered %d subjects", len(subjects))
    return subjects


def build_subject_filter_url(
    base_url: str,
    catoid: int,
    navoid: int,
    subject_code: str,
    *,
    cpage: int = 1,
    prefix_param: str = DEFAULT_PREFIX_PARAM,
) -> str:
    """Build the URL for a subject's course listing at a given page.

    Verified working param set for Acalog (catoid 19): the prefix filter
    (filter[27]=CS), the page index (filter[cpage]), and cur_cat_oid/search_database.
    """
    params = [
        ("catoid", str(catoid)),
        ("navoid", str(navoid)),
        (prefix_param, subject_code),
        ("filter[cpage]", str(cpage)),
        ("cur_cat_oid", str(catoid)),
        ("expand", ""),
        ("search_database", "Filter"),
    ]
    return urljoin(base_url, f"/content.php?{urlencode(params)}")
