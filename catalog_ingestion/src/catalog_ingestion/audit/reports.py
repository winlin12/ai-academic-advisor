"""Generate human-readable audit reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from catalog_ingestion.audit.validation import ValidationResult
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
