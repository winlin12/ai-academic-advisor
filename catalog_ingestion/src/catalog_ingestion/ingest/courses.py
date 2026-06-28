"""Ingest parsed course data into the database. Idempotent via upsert."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from catalog_ingestion.db.models import (
    CatalogYear,
    Course,
    PrerequisiteRule,
    SourcePage,
    Subject,
)
from catalog_ingestion.fetch.cache import content_hash
from catalog_ingestion.parse.courses import ParsedCourse
from catalog_ingestion.parse.prerequisites import parse_prerequisite_text

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.0"


def upsert_source_page(
    session: Session,
    *,
    url: str,
    html: str,
    http_status: int,
    catalog_year_id: uuid.UUID | None,
    page_type: str,
) -> SourcePage:
    """Insert or retrieve a source page, updating if content changed."""
    h = content_hash(html)
    existing = session.query(SourcePage).filter_by(url=url, content_hash=h).first()
    if existing:
        return existing

    page = SourcePage(
        url=url,
        catalog_year_id=catalog_year_id,
        page_type=page_type,
        http_status=http_status,
        content_hash=h,
        raw_html=html,
        raw_text=None,
        parser_version=PARSER_VERSION,
        fetched_at=datetime.utcnow(),
    )
    session.add(page)
    session.flush()
    return page


def get_or_create_subject(
    session: Session,
    *,
    catalog_year_id: uuid.UUID,
    subject_code: str,
) -> Subject:
    existing = (
        session.query(Subject)
        .filter_by(catalog_year_id=catalog_year_id, code=subject_code)
        .first()
    )
    if existing:
        return existing
    subj = Subject(catalog_year_id=catalog_year_id, code=subject_code)
    session.add(subj)
    session.flush()
    return subj


def ingest_course(
    session: Session,
    *,
    parsed: ParsedCourse,
    catalog_year_id: uuid.UUID,
    source_page_id: uuid.UUID | None,
) -> Course:
    """Insert or update a course record. Idempotent."""
    subject = get_or_create_subject(
        session, catalog_year_id=catalog_year_id, subject_code=parsed.subject_code
    )

    existing = (
        session.query(Course)
        .filter_by(catalog_year_id=catalog_year_id, course_code=parsed.course_code)
        .first()
    )

    if existing:
        existing.title = parsed.title
        existing.description = parsed.description
        existing.credit_hours_min = parsed.credit_hours_min
        existing.credit_hours_max = parsed.credit_hours_max
        existing.credit_hours_raw = parsed.credit_hours_raw
        existing.prerequisites_raw = parsed.prerequisites_raw
        existing.corequisites_raw = parsed.corequisites_raw
        existing.restrictions_raw = parsed.restrictions_raw
        existing.attributes_raw = parsed.attributes_raw
        existing.source_page_id = source_page_id
        existing.coid = parsed.coid
        session.flush()
        course = existing
    else:
        course = Course(
            catalog_year_id=catalog_year_id,
            subject_id=subject.id,
            subject_code=parsed.subject_code,
            course_number=parsed.course_number,
            course_code=parsed.course_code,
            coid=parsed.coid,
            title=parsed.title,
            description=parsed.description,
            credit_hours_min=parsed.credit_hours_min,
            credit_hours_max=parsed.credit_hours_max,
            credit_hours_raw=parsed.credit_hours_raw,
            prerequisites_raw=parsed.prerequisites_raw,
            corequisites_raw=parsed.corequisites_raw,
            restrictions_raw=parsed.restrictions_raw,
            attributes_raw=parsed.attributes_raw,
            source_page_id=source_page_id,
        )
        session.add(course)
        session.flush()

    if parsed.prerequisites_raw:
        _upsert_prerequisite_rule(session, course=course, raw_text=parsed.prerequisites_raw)

    if parsed.warnings:
        logger.warning("Warnings for %s: %s", parsed.course_code, "; ".join(parsed.warnings))

    return course


def _upsert_prerequisite_rule(session: Session, *, course: Course, raw_text: str) -> None:
    existing = (
        session.query(PrerequisiteRule)
        .filter_by(course_id=course.id, raw_text=raw_text)
        .first()
    )
    if existing:
        return

    result = parse_prerequisite_text(raw_text)

    rule = PrerequisiteRule(
        course_id=course.id,
        raw_text=result.raw_text,
        parsed_json=result.parsed_json,
        parse_confidence=result.parse_confidence,
        parser_notes=result.parser_notes,
    )
    session.add(rule)
    session.flush()


def ingest_course_batch(
    session: Session,
    *,
    parsed_courses: list[ParsedCourse],
    catalog_year: CatalogYear,
    html_map: dict[str, str],
) -> dict[str, str]:
    """Ingest a batch of courses. Returns {course_code: 'inserted'|'updated'|'skipped'}."""
    stats: dict[str, str] = {}

    for parsed in parsed_courses:
        try:
            source_page_id: uuid.UUID | None = None
            html = html_map.get(parsed.source_url)
            if html:
                sp = upsert_source_page(
                    session,
                    url=parsed.source_url,
                    html=html,
                    http_status=200,
                    catalog_year_id=catalog_year.id,
                    page_type="course",
                )
                source_page_id = sp.id

            ingest_course(
                session,
                parsed=parsed,
                catalog_year_id=catalog_year.id,
                source_page_id=source_page_id,
            )
            stats[parsed.course_code] = "ok"
            logger.debug("Ingested course %s", parsed.course_code)
        except Exception as exc:
            logger.error("Failed to ingest course %s: %s", parsed.course_code, exc)
            stats[parsed.course_code] = f"error: {exc}"
            session.rollback()

    return stats
