"""Import course/subject data from the PurdueIO (Purdue.io) OData API.

PurdueIO (https://api.purdue.io/odata) is the canonical source for Purdue's **course
catalog** — subject codes, course numbers, titles, credit hours, descriptions. It does
NOT contain degree programs or requirements; those come from scraping the Acalog catalog
(see ingest/programs.py). This is the intended division of labour:

    courses/subjects  <- PurdueIO API     (this module)
    programs/requirements <- catalog scrape (ingest/programs.py)
    requirement_options.course_id  <- resolved by course code (SUBJECT NUMBER)

The API enforces server-driven paging and rejects ``$top`` (MaxTop=0), so we follow
``@odata.nextLink`` and never send ``$top``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from sqlalchemy.orm import Session

from catalog_ingestion.db.models import (
    CatalogYear,
    Course,
    Program,
    PurdueapiCoursesStaging,
    PurdueapiSubjectsStaging,
    PurdueapiTermsStaging,
    RequirementGroup,
    RequirementOption,
)
from catalog_ingestion.ingest.courses import get_or_create_subject

logger = logging.getLogger(__name__)

DEFAULT_ODATA_URL = "https://api.purdue.io/odata"


def _fetch_odata(url: str) -> dict[str, Any]:
    with httpx.Client(timeout=120, headers={"Accept": "application/json"}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def _fetch_all(odata_url: str, entity: str, query: str = "") -> list[dict[str, Any]]:
    """Fetch every row of an OData entity, following @odata.nextLink. No $top."""
    items: list[dict[str, Any]] = []
    url: str | None = f"{odata_url}/{entity}"
    if query:
        url += f"?{query}"
    while url:
        data = _fetch_odata(url)
        items.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return items


def fetch_subject_index(odata_url: str = DEFAULT_ODATA_URL) -> dict[str, dict[str, str]]:
    """Return {SubjectId: {'abbr': 'CS', 'name': 'Computer Science'}}."""
    index: dict[str, dict[str, str]] = {}
    for item in _fetch_all(odata_url, "Subjects"):
        sid = str(item.get("Id", ""))
        if sid:
            index[sid] = {
                "abbr": (item.get("Abbreviation") or "").strip().upper(),
                "name": item.get("Name") or "",
            }
    return index


# ---------------------------------------------------------------------------
# Primary path: load PurdueIO courses into the real subjects/courses tables
# ---------------------------------------------------------------------------

def import_courses_to_catalog(
    session: Session,
    *,
    catalog_year: CatalogYear,
    odata_url: str = DEFAULT_ODATA_URL,
    subject_codes: list[str] | None = None,
) -> dict[str, int]:
    """Load PurdueIO subjects + courses into the catalog_ingestion course tables.

    Courses are keyed by (catalog_year_id, course_code) and deduplicated by code
    (preferring an entry that has a description). PurdueIO has no Acalog coid, so
    requirement options link to these courses by code via relink (below).
    """
    wanted = {c.strip().upper() for c in subject_codes} if subject_codes else None
    subject_index = fetch_subject_index(odata_url)

    # SubjectIds we care about (for client-side filtering — the API returns all courses
    # in one page and we don't rely on $filter).
    wanted_subject_ids: set[str] | None = None
    if wanted is not None:
        wanted_subject_ids = {
            sid for sid, meta in subject_index.items() if meta["abbr"] in wanted
        }

    # Ensure subjects exist (and carry names) for the catalog year.
    subject_cache: dict[str, Any] = {}
    for sid, meta in subject_index.items():
        abbr = meta["abbr"]
        if not abbr or (wanted is not None and abbr not in wanted):
            continue
        subj = get_or_create_subject(
            session, catalog_year_id=catalog_year.id, subject_code=abbr
        )
        if meta["name"] and not subj.name:
            subj.name = meta["name"]
        subject_cache[abbr] = subj
    session.flush()

    # Deduplicate PurdueIO courses by code, preferring rows with a description.
    best_by_code: dict[str, dict[str, Any]] = {}
    raw_courses = _fetch_all(odata_url, "Courses")
    for item in raw_courses:
        sid = str(item.get("SubjectId", ""))
        if wanted_subject_ids is not None and sid not in wanted_subject_ids:
            continue
        meta = subject_index.get(sid)
        if not meta or not meta["abbr"]:
            continue
        number = (item.get("Number") or "").strip()
        if not number:
            continue
        code = f"{meta['abbr']} {number}"
        prev = best_by_code.get(code)
        if prev is None or (not prev.get("Description") and item.get("Description")):
            item = {**item, "_abbr": meta["abbr"], "_number": number}
            best_by_code[code] = item

    inserted = 0
    updated = 0
    for code, item in best_by_code.items():
        abbr = item["_abbr"]
        number = item["_number"]
        credits = item.get("CreditHours")
        subj = subject_cache.get(abbr) or get_or_create_subject(
            session, catalog_year_id=catalog_year.id, subject_code=abbr
        )
        existing = (
            session.query(Course)
            .filter_by(catalog_year_id=catalog_year.id, course_code=code)
            .first()
        )
        if existing:
            existing.subject_id = subj.id
            existing.title = item.get("Title") or existing.title
            existing.description = item.get("Description") or existing.description
            if credits is not None:
                existing.credit_hours_min = credits
                existing.credit_hours_max = credits
            updated += 1
        else:
            session.add(
                Course(
                    catalog_year_id=catalog_year.id,
                    subject_id=subj.id,
                    subject_code=abbr,
                    course_number=number,
                    course_code=code,
                    coid=None,
                    title=item.get("Title"),
                    description=item.get("Description") or None,
                    credit_hours_min=credits,
                    credit_hours_max=credits,
                    credit_hours_raw=(f"{credits}" if credits is not None else None),
                    source_page_id=None,
                )
            )
            inserted += 1
    session.flush()

    relinked = relink_requirement_options_by_code(session, catalog_year_id=catalog_year.id)

    logger.info(
        "PurdueIO -> catalog: %d inserted, %d updated, %d requirement options linked",
        inserted, updated, relinked,
    )
    return {"inserted": inserted, "updated": updated, "relinked": relinked}


def relink_requirement_options_by_code(
    session: Session, *, catalog_year_id: uuid.UUID
) -> int:
    """Resolve requirement_options.course_id by matching course_code_raw to a course.

    Run after courses are (re)loaded so requirement lists point at real course rows.
    """
    code_map: dict[str, uuid.UUID] = dict(
        session.query(Course.course_code, Course.id)
        .filter(Course.catalog_year_id == catalog_year_id)
        .all()
    )
    if not code_map:
        return 0

    options = (
        session.query(RequirementOption)
        .join(RequirementGroup, RequirementOption.requirement_group_id == RequirementGroup.id)
        .join(Program, RequirementGroup.program_id == Program.id)
        .filter(
            Program.catalog_year_id == catalog_year_id,
            RequirementOption.course_id.is_(None),
            RequirementOption.course_code_raw.isnot(None),
        )
        .all()
    )
    linked = 0
    for opt in options:
        course_id = code_map.get(opt.course_code_raw)
        if course_id:
            opt.course_id = course_id
            linked += 1
    session.flush()
    return linked


# ---------------------------------------------------------------------------
# Secondary path: staging tables (cross-reference / data-quality comparison)
# ---------------------------------------------------------------------------

def import_subjects(
    session: Session,
    *,
    odata_url: str = DEFAULT_ODATA_URL,
    subject_codes: list[str] | None = None,
) -> int:
    """Import subjects into the staging table. Returns record count."""
    session.query(PurdueapiSubjectsStaging).delete()
    wanted = {c.upper() for c in subject_codes} if subject_codes else None
    count = 0
    for item in _fetch_all(odata_url, "Subjects"):
        abbr = (item.get("Abbreviation") or "").upper()
        if wanted and abbr not in wanted:
            continue
        session.add(
            PurdueapiSubjectsStaging(
                purdueapi_id=str(item.get("Id", "")),
                abbreviation=abbr,
                name=item.get("Name"),
                raw_json=item,
            )
        )
        count += 1
    session.flush()
    logger.info("Imported %d subjects from PurdueIO (staging)", count)
    return count


def import_courses(
    session: Session,
    *,
    odata_url: str = DEFAULT_ODATA_URL,
    subject_codes: list[str] | None = None,
) -> int:
    """Import courses into the staging table. Returns record count."""
    session.query(PurdueapiCoursesStaging).delete()
    wanted = {c.upper() for c in subject_codes} if subject_codes else None
    subject_index = fetch_subject_index(odata_url)

    count = 0
    for item in _fetch_all(odata_url, "Courses"):
        meta = subject_index.get(str(item.get("SubjectId", "")))
        abbr = meta["abbr"] if meta else None
        if wanted and (abbr or "") not in wanted:
            continue
        session.add(
            PurdueapiCoursesStaging(
                purdueapi_id=str(item.get("Id", "")),
                subject_abbreviation=abbr,
                number=item.get("Number"),
                title=item.get("Title"),
                credit_hours=item.get("CreditHours"),
                description=item.get("Description"),
                raw_json=item,
            )
        )
        count += 1
    session.flush()
    logger.info("Imported %d courses from PurdueIO (staging)", count)
    return count


def import_terms(session: Session, *, odata_url: str = DEFAULT_ODATA_URL) -> int:
    """Import terms into the staging table. Returns record count."""
    session.query(PurdueapiTermsStaging).delete()
    count = 0
    for item in _fetch_all(odata_url, "Terms", query="$orderby=Code desc"):
        session.add(
            PurdueapiTermsStaging(
                purdueapi_id=str(item.get("Id", "")),
                code=item.get("Code"),
                name=item.get("Name"),
                start_date=str(item.get("StartDate", "") or ""),
                end_date=str(item.get("EndDate", "") or ""),
                raw_json=item,
            )
        )
        count += 1
    session.flush()
    logger.info("Imported %d terms from PurdueIO (staging)", count)
    return count


def compare_courses_with_catalog(
    session: Session,
    *,
    catalog_year_label: str,
    subject_code: str | None = None,
) -> list[dict[str, Any]]:
    """Compare PurdueIO staging course data vs catalog course rows by code."""
    year = session.query(CatalogYear).filter_by(label=catalog_year_label).first()
    if not year:
        logger.warning("Catalog year %r not found", catalog_year_label)
        return []

    mismatches: list[dict[str, Any]] = []
    catalog_q = session.query(Course).filter_by(catalog_year_id=year.id)
    if subject_code:
        catalog_q = catalog_q.filter_by(subject_code=subject_code)

    for course in catalog_q:
        code = course.course_code
        subject, number = code.split(" ", 1) if " " in code else (code, "")
        api_course = (
            session.query(PurdueapiCoursesStaging)
            .filter_by(subject_abbreviation=subject, number=number)
            .first()
        )
        if not api_course:
            mismatches.append({
                "course_code": code,
                "issue": "missing_in_purdueapi",
                "catalog_title": course.title,
                "purdueapi_title": None,
            })
            continue
        if course.title and api_course.title and course.title != api_course.title:
            mismatches.append({
                "course_code": code,
                "issue": "title_mismatch",
                "catalog_value": course.title,
                "purdueapi_value": api_course.title,
            })

    logger.info("Found %d mismatches for %s (subject=%s)", len(mismatches), catalog_year_label, subject_code)
    return mismatches
