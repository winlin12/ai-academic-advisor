"""Import course/subject/term data from PurdueAPI OData endpoint.

PurdueAPI (Purdue.io) is an open-source project that provides scheduling and
course data via OData. We treat it as a supporting source, not a competitor.

This module imports into staging tables (purdueapi_*_staging) so it can be
compared against scraped catalog data without overwriting it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from sqlalchemy.orm import Session

from catalog_ingestion.db.models import (
    PurdueapiCoursesStaging,
    PurdueapiSubjectsStaging,
    PurdueapiTermsStaging,
)

logger = logging.getLogger(__name__)

DEFAULT_ODATA_URL = "https://api.purdue.io/odata"
PAGE_SIZE = 200


def import_subjects(
    session: Session,
    *,
    odata_url: str = DEFAULT_ODATA_URL,
    subject_codes: list[str] | None = None,
) -> int:
    """Import subjects from PurdueAPI into staging table. Returns record count."""
    session.query(PurdueapiSubjectsStaging).delete()

    count = 0
    url: str | None = f"{odata_url}/Subjects?$top={PAGE_SIZE}"
    while url:
        data = _fetch_odata(url)
        for item in data.get("value", []):
            abbr = item.get("Abbreviation", "")
            if subject_codes and abbr not in subject_codes:
                continue
            staging = PurdueapiSubjectsStaging(
                purdueapi_id=str(item.get("Id", "")),
                abbreviation=abbr,
                name=item.get("Name"),
                raw_json=item,
            )
            session.add(staging)
            count += 1
        url = data.get("@odata.nextLink")

    session.flush()
    logger.info("Imported %d subjects from PurdueAPI", count)
    return count


def import_courses(
    session: Session,
    *,
    odata_url: str = DEFAULT_ODATA_URL,
    subject_codes: list[str] | None = None,
) -> int:
    """Import courses from PurdueAPI into staging table. Returns record count."""
    session.query(PurdueapiCoursesStaging).delete()

    count = 0
    filter_param = ""
    if subject_codes:
        quoted = ", ".join(f"'{c}'" for c in subject_codes)
        filter_param = f"&$filter=Subject/Abbreviation in ({quoted})"

    url: str | None = (
        f"{odata_url}/Courses?$expand=Subject&$top={PAGE_SIZE}{filter_param}"
    )
    while url:
        data = _fetch_odata(url)
        for item in data.get("value", []):
            subject = item.get("Subject") or {}
            staging = PurdueapiCoursesStaging(
                purdueapi_id=str(item.get("Id", "")),
                subject_abbreviation=subject.get("Abbreviation"),
                number=item.get("Number"),
                title=item.get("Title"),
                credit_hours=item.get("CreditHours"),
                description=item.get("Description"),
                raw_json=item,
            )
            session.add(staging)
            count += 1
        url = data.get("@odata.nextLink")

    session.flush()
    logger.info("Imported %d courses from PurdueAPI", count)
    return count


def import_terms(
    session: Session,
    *,
    odata_url: str = DEFAULT_ODATA_URL,
) -> int:
    """Import terms from PurdueAPI into staging table. Returns record count."""
    session.query(PurdueapiTermsStaging).delete()

    count = 0
    url: str | None = f"{odata_url}/Terms?$top={PAGE_SIZE}&$orderby=Code desc"
    while url:
        data = _fetch_odata(url)
        for item in data.get("value", []):
            staging = PurdueapiTermsStaging(
                purdueapi_id=str(item.get("Id", "")),
                code=item.get("Code"),
                name=item.get("Name"),
                start_date=str(item.get("StartDate", "") or ""),
                end_date=str(item.get("EndDate", "") or ""),
                raw_json=item,
            )
            session.add(staging)
            count += 1
        url = data.get("@odata.nextLink")

    session.flush()
    logger.info("Imported %d terms from PurdueAPI", count)
    return count


def _fetch_odata(url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=60, headers={"Accept": "application/json"}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error("PurdueAPI fetch failed for %s: %s", url, exc)
        return {"value": []}


def compare_courses_with_catalog(
    session: Session,
    *,
    catalog_year_label: str,
    subject_code: str | None = None,
) -> list[dict[str, Any]]:
    """Compare PurdueAPI course staging data vs scraped catalog courses.

    Returns list of mismatch records with fields:
      course_code, field, catalog_value, purdueapi_value
    """
    from catalog_ingestion.db.models import CatalogYear, Course

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
