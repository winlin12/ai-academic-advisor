"""Ingest parsed program and requirement data into the database. Idempotent."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from catalog_ingestion.db.models import (
    CatalogYear,
    College,
    Program,
    ProgramNote,
    RequirementGroup,
    RequirementOption,
)
from catalog_ingestion.ingest.courses import upsert_source_page
from catalog_ingestion.parse.programs import ParsedProgram
from catalog_ingestion.parse.requirements import RequirementGroupData, RequirementOptionData

logger = logging.getLogger(__name__)


def get_or_create_college(
    session: Session,
    *,
    catalog_year_id: uuid.UUID,
    name: str,
) -> College:
    existing = (
        session.query(College)
        .filter_by(catalog_year_id=catalog_year_id, name=name)
        .first()
    )
    if existing:
        return existing
    college = College(catalog_year_id=catalog_year_id, name=name)
    session.add(college)
    session.flush()
    return college


def ingest_program(
    session: Session,
    *,
    parsed: ParsedProgram,
    requirement_groups: list[RequirementGroupData],
    catalog_year: CatalogYear,
    html: str,
) -> Program:
    """Insert or update a program and its requirements. Idempotent."""
    source_page_id: uuid.UUID | None = None
    if html:
        sp = upsert_source_page(
            session,
            url=parsed.source_url,
            html=html,
            http_status=200,
            catalog_year_id=catalog_year.id,
            page_type="program",
        )
        source_page_id = sp.id

    college_id: uuid.UUID | None = None
    if parsed.college_name:
        college = get_or_create_college(
            session,
            catalog_year_id=catalog_year.id,
            name=parsed.college_name,
        )
        college_id = college.id

    existing = (
        session.query(Program)
        .filter_by(
            catalog_year_id=catalog_year.id,
            name=parsed.name,
            degree_type=parsed.degree_type,
        )
        .first()
    )

    if existing:
        existing.program_type = parsed.program_type
        existing.campus = parsed.campus
        existing.college_id = college_id
        existing.total_credits_raw = parsed.total_credits_raw
        existing.total_credits_min = parsed.total_credits_min
        existing.description = parsed.description
        existing.poid = parsed.poid
        existing.source_page_id = source_page_id
        session.flush()
        program = existing
        # Clear and re-insert requirements
        session.query(RequirementGroup).filter_by(program_id=program.id).delete()
        session.query(ProgramNote).filter_by(program_id=program.id).delete()
    else:
        program = Program(
            catalog_year_id=catalog_year.id,
            college_id=college_id,
            name=parsed.name,
            degree_type=parsed.degree_type,
            program_type=parsed.program_type,
            campus=parsed.campus,
            total_credits_raw=parsed.total_credits_raw,
            total_credits_min=parsed.total_credits_min,
            description=parsed.description,
            poid=parsed.poid,
            source_page_id=source_page_id,
        )
        session.add(program)
        session.flush()

    for group_data in requirement_groups:
        _ingest_requirement_group(
            session,
            group_data=group_data,
            program=program,
            catalog_year=catalog_year,
            source_page_id=source_page_id,
            parent_group_id=None,
        )

    if parsed.warnings:
        for warning in parsed.warnings:
            session.add(
                ProgramNote(
                    program_id=program.id,
                    note_text=warning,
                    note_type="parser_warning",
                    source_page_id=source_page_id,
                )
            )
        session.flush()

    logger.debug("Ingested program: %s (%s)", parsed.name, parsed.degree_type)
    return program


def _ingest_requirement_group(
    session: Session,
    *,
    group_data: RequirementGroupData,
    program: Program,
    catalog_year: CatalogYear,
    source_page_id: uuid.UUID | None,
    parent_group_id: uuid.UUID | None,
) -> RequirementGroup:
    group = RequirementGroup(
        program_id=program.id,
        parent_group_id=parent_group_id,
        name=group_data.name,
        requirement_type=group_data.requirement_type,
        credits_min=group_data.credits_min,
        credits_max=group_data.credits_max,
        raw_text=group_data.raw_text,
        display_order=group_data.display_order,
        source_page_id=source_page_id,
    )
    session.add(group)
    session.flush()

    for opt_data in group_data.options:
        _ingest_requirement_option(
            session,
            opt_data=opt_data,
            group=group,
            catalog_year=catalog_year,
        )

    for child in group_data.children:
        _ingest_requirement_group(
            session,
            group_data=child,
            program=program,
            catalog_year=catalog_year,
            source_page_id=source_page_id,
            parent_group_id=group.id,
        )

    return group


def _ingest_requirement_option(
    session: Session,
    *,
    opt_data: RequirementOptionData,
    group: RequirementGroup,
    catalog_year: CatalogYear,
) -> RequirementOption:
    from catalog_ingestion.db.models import Course

    # Resolve the course FK by stable Acalog coid first (most reliable), then fall back
    # to the printed course code. course_id stays null if the course is not yet scraped;
    # coid/course_code_raw are always preserved so it can be backfilled later.
    course_id: uuid.UUID | None = None
    if opt_data.coid is not None:
        course = (
            session.query(Course)
            .filter_by(catalog_year_id=catalog_year.id, coid=opt_data.coid)
            .first()
        )
        if course:
            course_id = course.id
    if course_id is None and opt_data.course_code_raw:
        course = (
            session.query(Course)
            .filter_by(
                catalog_year_id=catalog_year.id,
                course_code=opt_data.course_code_raw,
            )
            .first()
        )
        if course:
            course_id = course.id

    option = RequirementOption(
        requirement_group_id=group.id,
        course_id=course_id,
        coid=opt_data.coid,
        course_code_raw=opt_data.course_code_raw,
        course_title_raw=opt_data.title,
        option_text=opt_data.option_text,
        credits=opt_data.credits,
        credits_max=opt_data.credits_max,
        minimum_grade=opt_data.minimum_grade,
        is_required=not opt_data.is_selective,
        is_selective_option=opt_data.is_selective,
        display_order=opt_data.display_order,
    )
    session.add(option)
    session.flush()
    return option
