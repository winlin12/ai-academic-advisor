"""Generate human-readable audit reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from catalog_ingestion.audit.validation import ValidationResult, validate_catalog_year
from catalog_ingestion.db.models import CatalogYear, Course, Program, Subject


def report_catalog_summary(session: Session, *, catalog_year_label: str) -> dict[str, Any]:
    """Return a summary dict of what's been ingested for a catalog year."""
    year = session.query(CatalogYear).filter_by(label=catalog_year_label).first()
    if not year:
        return {"error": f"Catalog year {catalog_year_label!r} not found"}

    course_count = session.query(Course).filter_by(catalog_year_id=year.id).count()
    program_count = session.query(Program).filter_by(catalog_year_id=year.id).count()
    subject_count = session.query(Subject).filter_by(catalog_year_id=year.id).count()

    subjects = session.query(Subject).filter_by(catalog_year_id=year.id).all()
    courses_per_subject: dict[str, int] = {}
    for subj in subjects:
        cnt = session.query(Course).filter_by(
            catalog_year_id=year.id, subject_code=subj.code
        ).count()
        if cnt > 0:
            courses_per_subject[subj.code] = cnt

    return {
        "catalog_year": year.label,
        "catoid": year.catoid,
        "is_archived": year.is_archived,
        "subjects": subject_count,
        "courses": course_count,
        "programs": program_count,
        "courses_per_subject": courses_per_subject,
    }


def report_program(
    session: Session, *, catalog_year_label: str, program_name: str
) -> dict[str, Any]:
    """Return parsed requirement structure for a specific program."""
    from catalog_ingestion.db.models import RequirementGroup, RequirementOption

    year = session.query(CatalogYear).filter_by(label=catalog_year_label).first()
    if not year:
        return {"error": f"Catalog year {catalog_year_label!r} not found"}

    program = (
        session.query(Program)
        .filter(
            Program.catalog_year_id == year.id,
            Program.name.ilike(f"%{program_name}%"),
        )
        .first()
    )
    if not program:
        return {"error": f"Program matching {program_name!r} not found in {catalog_year_label}"}

    groups = (
        session.query(RequirementGroup)
        .filter_by(program_id=program.id, parent_group_id=None)
        .order_by(RequirementGroup.display_order)
        .all()
    )

    def group_to_dict(g: RequirementGroup) -> dict[str, Any]:
        opts = (
            session.query(RequirementOption)
            .filter_by(requirement_group_id=g.id)
            .order_by(RequirementOption.display_order)
            .all()
        )
        children = (
            session.query(RequirementGroup)
            .filter_by(parent_group_id=g.id)
            .order_by(RequirementGroup.display_order)
            .all()
        )
        return {
            "name": g.name,
            "type": g.requirement_type,
            "credits_min": g.credits_min,
            "credits_max": g.credits_max,
            "options": [
                {
                    "course_code": o.course_code_raw,
                    "text": o.option_text,
                    "is_selective": o.is_selective_option,
                    "credits": o.credits,
                }
                for o in opts
            ],
            "children": [group_to_dict(c) for c in children],
        }

    return {
        "program": program.name,
        "degree_type": program.degree_type,
        "program_type": program.program_type,
        "campus": program.campus,
        "college": program.college.name if program.college else None,
        "total_credits": program.total_credits_min,
        "requirement_groups": [group_to_dict(g) for g in groups],
    }


def write_validation_report(
    results: list[ValidationResult],
    output_path: Path,
) -> None:
    """Write a human-readable validation report to a file."""
    lines: list[str] = [
        f"Validation Report — {results[0].catalog_year if results else 'unknown'}",
        "=" * 60,
    ]
    passed = sum(1 for r in results if r.passed)
    lines.append(f"\n{passed}/{len(results)} checks passed\n")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.check_name}: {r.notes}")
        if not r.passed and r.details:
            for detail in r.details[:5]:
                lines.append(f"       {detail}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
