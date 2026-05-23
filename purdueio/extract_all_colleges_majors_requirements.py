#!/usr/bin/env python3
"""Extract 2024 requirements for all majors across all college PDFs.

Pipeline:
1) discover college/school combined PDFs
2) list majors per college
3) extract requirement blocks for each major

Examples:
  python3 purdueio/extract_all_colleges_majors_requirements.py
  python3 purdueio/extract_all_colleges_majors_requirements.py --school "College of Science"
  python3 purdueio/extract_all_colleges_majors_requirements.py --max-majors-per-school 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_major_requirements import (
    choose_heading,
    extract_blocks,
    extract_lines,
    find_degree_headings,
    find_section_end_line,
    slugify as major_slugify,
    write_markdown as write_requirements_markdown,
)
from list_school_majors import (
    DegreeHeading,
    choose_pdf_for_school,
    extract_degree_headings,
    find_year_combined_pdfs,
    render_school_name_from_file,
    slugify as school_slugify,
)

DEFAULT_OUTPUT_ROOT = Path("purdueio/data/extracted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract major requirements for all colleges/schools in a catalog year."
    )
    parser.add_argument("--year", type=int, default=2024, help="Catalog year (default: 2024).")
    parser.add_argument(
        "--school",
        action="append",
        default=[],
        help="Limit to one or more schools, e.g. --school 'College of Science'.",
    )
    parser.add_argument(
        "--include-graduate",
        action="store_true",
        help="Include graduate degree headings when listing majors.",
    )
    parser.add_argument(
        "--max-majors-per-school",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory (default: purdueio/data/extracted).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if requirement files already exist.",
    )
    return parser.parse_args()


def write_school_majors_files(
    school_dir: Path,
    year: int,
    school: str,
    pdf_path: Path,
    majors: list[DegreeHeading],
) -> tuple[Path, Path]:
    majors_json = school_dir / "majors.json"
    majors_md = school_dir / "majors.md"

    payload = {
        "year": year,
        "school": school,
        "source_pdf": str(pdf_path),
        "total_majors": len(majors),
        "majors": [
            {
                "title": m.title,
                "degree": m.degree,
                "variant": m.variant,
                "page": m.page,
                "heading": m.raw_heading,
            }
            for m in majors
        ],
    }

    majors_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# {school} Majors ({year})",
        "",
        f"- Source PDF: `{pdf_path}`",
        f"- Total majors found: {len(majors)}",
        "",
    ]
    for idx, major in enumerate(majors, start=1):
        suffix = f" ({major.variant})" if major.variant else ""
        md_lines.append(
            f"{idx}. {major.title}, {major.degree}{suffix} (page {major.page})"
        )

    majors_md.write_text("\n".join(md_lines), encoding="utf-8")
    return majors_json, majors_md


def extract_single_major_requirement(
    *,
    lines,
    headings,
    major: DegreeHeading,
    pdf_path: Path,
) -> tuple[dict[str, object], str | None]:
    try:
        selected, candidate_list = choose_heading(
            headings=headings,
            major=major.title,
            degree=major.degree,
            variant=major.variant,
            concentration=None,
        )
    except RuntimeError as exc:
        return {}, str(exc)

    end_line = find_section_end_line(lines, selected, headings)
    section_lines = [rec for rec in lines if selected.line_index <= rec.idx < end_line]
    blocks = extract_blocks(section_lines)

    section_start_page = section_lines[0].page if section_lines else selected.page
    section_end_page = section_lines[-1].page if section_lines else selected.page

    output = {
        "source_pdf": str(pdf_path),
        "target": {
            "major": major.title,
            "degree": major.degree,
            "variant": major.variant,
            "concentration": None,
        },
        "selected_heading": selected.line_text,
        "section_pages": {
            "start": section_start_page,
            "end_inclusive": section_end_page,
        },
        "candidate_headings": [
            {
                "page": c.page,
                "heading": c.line_text,
                "score": c.score,
            }
            for c in candidate_list[:8]
        ],
        "major_requirement_blocks": blocks,
    }

    return output, None


def school_to_pdf_list(year: int, schools: list[str]) -> list[tuple[str, Path]]:
    if schools:
        selected: list[tuple[str, Path]] = []
        for school in schools:
            pdf_path = choose_pdf_for_school(year, school)
            selected.append((school, pdf_path))
        return selected

    return [
        (render_school_name_from_file(path, year), path)
        for path in find_year_combined_pdfs(year)
    ]


def write_run_summary(summary_path: Path, summary: dict[str, object]) -> Path:
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = summary_path.with_suffix(".md")
    lines: list[str] = []
    lines.append(f"# Extraction Summary ({summary['year']})")
    lines.append("")
    lines.append(f"- Schools processed: {summary['schools_processed']}")
    lines.append(f"- Majors discovered: {summary['majors_discovered']}")
    lines.append(f"- Requirements extracted: {summary['requirements_extracted']}")
    lines.append(f"- Requirements skipped: {summary['requirements_skipped']}")
    lines.append(f"- Failures: {summary['requirements_failed']}")
    lines.append("")

    for school in summary["schools"]:
        lines.append(f"## {school['school']}")
        lines.append("")
        lines.append(f"- Source PDF: `{school['source_pdf']}`")
        lines.append(f"- Majors: {school['majors_total']}")
        lines.append(f"- Extracted: {school['requirements_extracted']}")
        lines.append(f"- Skipped: {school['requirements_skipped']}")
        lines.append(f"- Failed: {school['requirements_failed']}")
        failures = [m for m in school["majors"] if m.get("status") == "failed"]
        if failures:
            lines.append("- Failed majors:")
            for failure in failures:
                lines.append(f"  - {failure['title']} ({failure['degree']}): {failure.get('error', 'unknown error')}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    args = parse_args()

    base_dir = args.output_root / f"{args.year}_all_requirements"
    base_dir.mkdir(parents=True, exist_ok=True)

    schools = school_to_pdf_list(args.year, args.school)
    if not schools:
        print(f"No schools found for {args.year}")
        return 1

    run_summary: dict[str, object] = {
        "year": args.year,
        "include_graduate": args.include_graduate,
        "schools_processed": 0,
        "majors_discovered": 0,
        "requirements_extracted": 0,
        "requirements_skipped": 0,
        "requirements_failed": 0,
        "schools": [],
    }

    for school_name, pdf_path in schools:
        school_slug = school_slugify(school_name)
        school_dir = base_dir / school_slug
        req_dir = school_dir / "requirements"
        req_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {school_name} ===")
        print(f"PDF: {pdf_path}")

        majors = extract_degree_headings(pdf_path, include_graduate=args.include_graduate)
        if args.max_majors_per_school is not None:
            majors = majors[: args.max_majors_per_school]

        majors_json, majors_md = write_school_majors_files(
            school_dir=school_dir,
            year=args.year,
            school=school_name,
            pdf_path=pdf_path,
            majors=majors,
        )
        print(f"Majors: {len(majors)}")
        print(f"Wrote majors list: {majors_json}")

        lines = extract_lines(pdf_path)
        headings = find_degree_headings(lines)

        school_result: dict[str, object] = {
            "school": school_name,
            "source_pdf": str(pdf_path),
            "school_output_dir": str(school_dir),
            "majors_json": str(majors_json),
            "majors_md": str(majors_md),
            "majors_total": len(majors),
            "requirements_extracted": 0,
            "requirements_skipped": 0,
            "requirements_failed": 0,
            "majors": [],
        }

        for major in majors:
            major_slug = major_slugify(major.title)
            degree_slug = major_slugify(major.degree)
            variant_slug = major_slugify(major.variant) if major.variant else None
            stem = f"{major_slug}__{degree_slug}"
            if variant_slug:
                stem = f"{stem}__{variant_slug}"

            out_json = req_dir / f"{stem}.json"
            out_md = req_dir / f"{stem}.md"

            major_result = {
                "title": major.title,
                "degree": major.degree,
                "variant": major.variant,
                "page": major.page,
                "heading": major.raw_heading,
                "requirements_json": str(out_json),
                "requirements_md": str(out_md),
            }

            if not args.force and out_json.exists() and out_md.exists():
                major_result["status"] = "skipped"
                school_result["requirements_skipped"] += 1
                run_summary["requirements_skipped"] += 1
                school_result["majors"].append(major_result)
                continue

            payload, error = extract_single_major_requirement(
                lines=lines,
                headings=headings,
                major=major,
                pdf_path=pdf_path,
            )

            if error:
                major_result["status"] = "failed"
                major_result["error"] = error
                school_result["requirements_failed"] += 1
                run_summary["requirements_failed"] += 1
                school_result["majors"].append(major_result)
                print(f"[fail] {major.title}, {major.degree}: {error}")
                continue

            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            write_requirements_markdown(out_md, payload)

            major_result["status"] = "ok"
            major_result["parsed_blocks"] = len(payload.get("major_requirement_blocks", []))
            school_result["requirements_extracted"] += 1
            run_summary["requirements_extracted"] += 1
            school_result["majors"].append(major_result)
            print(
                f"[ok] {major.title}, {major.degree}"
                f" -> blocks={major_result['parsed_blocks']}"
            )

        run_summary["schools_processed"] += 1
        run_summary["majors_discovered"] += len(majors)
        run_summary["schools"].append(school_result)

    summary_json = base_dir / "summary.json"
    summary_md = write_run_summary(summary_json, run_summary)

    print("\n=== Done ===")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    print(
        "Totals: "
        f"schools={run_summary['schools_processed']}, "
        f"majors={run_summary['majors_discovered']}, "
        f"ok={run_summary['requirements_extracted']}, "
        f"skipped={run_summary['requirements_skipped']}, "
        f"failed={run_summary['requirements_failed']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
