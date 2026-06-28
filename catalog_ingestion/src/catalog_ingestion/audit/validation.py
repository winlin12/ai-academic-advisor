"""Validation checks for ingested catalog data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from catalog_ingestion.db.models import (
    CatalogYear,
    Course,
    PrerequisiteRule,
    Program,
    RequirementGroup,
    RequirementOption,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    check_name: str
    catalog_year: str
    passed: bool
    count: int
    details: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def validate_catalog_year(
    session: Session, *, catalog_year_label: str
) -> list[ValidationResult]:
    """Run all validation checks for a catalog year. Returns list of results."""
    year = session.query(CatalogYear).filter_by(label=catalog_year_label).first()
    if not year:
        return [
            ValidationResult(
                check_name="catalog_year_exists",
                catalog_year=catalog_year_label,
                passed=False,
                count=0,
                notes=f"Catalog year {catalog_year_label!r} not found in database",
            )
        ]

    results: list[ValidationResult] = []
    results.append(_check_course_count(session, year))
    results.append(_check_program_count(session, year))
    results.append(_check_courses_with_no_title(session, year))
    results.append(_check_courses_with_no_description(session, year))
    results.append(_check_prerequisite_parse_failures(session, year))
    results.append(_check_requirement_groups_missing_raw(session, year))
    results.append(_check_programs_with_no_requirements(session, year))
    results.append(_check_orphan_requirement_options(session, year))
    return results


def _check_course_count(session: Session, year: CatalogYear) -> ValidationResult:
    count = session.query(Course).filter_by(catalog_year_id=year.id).count()
    return ValidationResult(
        check_name="course_count",
        catalog_year=year.label,
        passed=count > 0,
        count=count,
        notes=f"{count} courses for {year.label}",
    )


def _check_program_count(session: Session, year: CatalogYear) -> ValidationResult:
    count = session.query(Program).filter_by(catalog_year_id=year.id).count()
    return ValidationResult(
        check_name="program_count",
        catalog_year=year.label,
        passed=count > 0,
        count=count,
        notes=f"{count} programs for {year.label}",
    )


def _check_courses_with_no_title(session: Session, year: CatalogYear) -> ValidationResult:
    courses = (
        session.query(Course)
        .filter_by(catalog_year_id=year.id)
        .filter(Course.title.is_(None))
        .all()
    )
    return ValidationResult(
        check_name="courses_missing_title",
        catalog_year=year.label,
        passed=len(courses) == 0,
        count=len(courses),
        details=[{"course_code": c.course_code} for c in courses[:20]],
        notes=f"{len(courses)} courses missing title",
    )


def _check_courses_with_no_description(session: Session, year: CatalogYear) -> ValidationResult:
    courses = (
        session.query(Course)
        .filter_by(catalog_year_id=year.id)
        .filter(Course.description.is_(None))
        .all()
    )
    return ValidationResult(
        check_name="courses_missing_description",
        catalog_year=year.label,
        passed=True,
        count=len(courses),
        details=[{"course_code": c.course_code} for c in courses[:20]],
        notes=f"{len(courses)} courses missing description (informational)",
    )


def _check_prerequisite_parse_failures(session: Session, year: CatalogYear) -> ValidationResult:
    failed = (
        session.query(PrerequisiteRule)
        .join(Course)
        .filter(Course.catalog_year_id == year.id)
        .filter(PrerequisiteRule.parse_confidence == "low")
        .all()
    )
    return ValidationResult(
        check_name="prerequisite_parse_failures",
        catalog_year=year.label,
        passed=True,
        count=len(failed),
        details=[
            {"course_id": str(r.course_id), "raw": r.raw_text[:80], "notes": r.parser_notes}
            for r in failed[:20]
        ],
        notes=f"{len(failed)} prerequisites with low parse confidence (informational)",
    )


def _check_requirement_groups_missing_raw(
    session: Session, year: CatalogYear
) -> ValidationResult:
    groups = (
        session.query(RequirementGroup)
        .join(Program)
        .filter(Program.catalog_year_id == year.id)
        .filter(RequirementGroup.raw_text.is_(None))
        .all()
    )
    return ValidationResult(
        check_name="requirement_groups_missing_raw_text",
        catalog_year=year.label,
        passed=len(groups) == 0,
        count=len(groups),
        details=[{"group_id": str(g.id), "name": g.name} for g in groups[:20]],
        notes=f"{len(groups)} requirement groups with no raw_text",
    )


def _check_programs_with_no_requirements(
    session: Session, year: CatalogYear
) -> ValidationResult:
    programs_with_reqs = (
        session.query(Program.id)
        .join(RequirementGroup)
        .filter(Program.catalog_year_id == year.id)
        .distinct()
        .subquery()
    )
    programs_no_reqs = (
        session.query(Program)
        .filter_by(catalog_year_id=year.id)
        .filter(Program.id.not_in(programs_with_reqs))
        .all()
    )
    return ValidationResult(
        check_name="programs_without_requirements",
        catalog_year=year.label,
        passed=len(programs_no_reqs) == 0,
        count=len(programs_no_reqs),
        details=[{"program": p.name, "degree": p.degree_type} for p in programs_no_reqs[:20]],
        notes=f"{len(programs_no_reqs)} programs with no requirement groups",
    )


def _check_orphan_requirement_options(
    session: Session, year: CatalogYear
) -> ValidationResult:
    orphans = (
        session.query(RequirementOption)
        .join(RequirementGroup)
        .join(Program)
        .filter(Program.catalog_year_id == year.id)
        .filter(RequirementOption.course_id.is_(None))
        .filter(RequirementOption.course_code_raw.isnot(None))
        .all()
    )
    return ValidationResult(
        check_name="requirement_options_with_unresolved_course_code",
        catalog_year=year.label,
        passed=True,
        count=len(orphans),
        details=[
            {"course_code_raw": o.course_code_raw, "option_text": (o.option_text or "")[:60]}
            for o in orphans[:20]
        ],
        notes=f"{len(orphans)} requirement options with unresolved course_code_raw (informational)",
    )
