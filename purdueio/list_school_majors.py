#!/usr/bin/env python3
"""List all major headings for a given Purdue school/college PDF.

Examples:
  python3 purdueio/list_school_majors.py --school "College of Science"
  python3 purdueio/list_school_majors.py --school "College of Engineering" --year 2024
  python3 purdueio/list_school_majors.py --pdf purdueio/data/catalog_pdfs/2024-CollegeofScience-Combined.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pypdf'. Install with: python3 -m pip install --user pypdf"
    ) from exc

CATALOG_DIR = Path("purdueio/data/catalog_pdfs")
OUT_DIR = Path("purdueio/data/extracted")

# Examples matched:
#   Data Science, BS (CS)
#   Physics, BS
#   Business Analytics and Information Management, BSBA
HEADING_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9&'()/:,\.\- ]+?),\s*(?P<degree>[A-Z/]{2,8})(?:\s*\((?P<variant>[^)]+)\))?$"
)
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s*\d{3,5}\b")
SPLIT_HEADING_SUFFIX_RE = re.compile(
    r"^(Concentration|Track|Option),\s*[A-Z/]{2,8}(?:\s*\([^)]+\))?$", re.I
)


@dataclass(frozen=True)
class DegreeHeading:
    title: str
    degree: str
    variant: str | None
    page: int
    raw_heading: str


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    return " ".join(value.split()).strip()


def normalize_for_match(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return " ".join(value.split())


def normalize_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def slugify(value: str) -> str:
    text = normalize_for_match(value)
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "school"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List all majors for a given school/college.")
    parser.add_argument("--school", default=None, help="School name, e.g. 'College of Science'.")
    parser.add_argument("--year", type=int, default=2024, help="Catalog year prefix (default: 2024).")
    parser.add_argument("--pdf", type=Path, default=None, help="Optional explicit PDF path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional output markdown path.")
    parser.add_argument(
        "--list-schools",
        action="store_true",
        help="List available combined PDFs for the selected year and exit.",
    )
    parser.add_argument(
        "--include-graduate",
        action="store_true",
        help="Include non-bachelor degree headings (MS, PhD, etc.).",
    )
    return parser.parse_args()


def find_year_combined_pdfs(year: int) -> list[Path]:
    return sorted(CATALOG_DIR.glob(f"{year}-*Combined.pdf"))


def render_school_name_from_file(path: Path, year: int) -> str:
    name = path.stem
    if name.startswith(f"{year}-"):
        name = name[len(f"{year}-") :]
    if name.endswith("-Combined"):
        name = name[: -len("-Combined")]
    name = name.replace("-", " ")
    name = re.sub(r"Collegeof", "College of ", name, flags=re.I)
    name = re.sub(r"Schoolof", "School of ", name, flags=re.I)
    name = re.sub(r"Purduein", "Purdue in ", name, flags=re.I)
    name = re.sub(r"Librariesand", "Libraries and ", name, flags=re.I)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return normalize_text(name)


def choose_pdf_for_school(year: int, school: str) -> Path:
    candidates = find_year_combined_pdfs(year)
    if not candidates:
        raise RuntimeError(f"No combined PDFs found for year {year} in {CATALOG_DIR}")

    school_norm = normalize_for_match(school)
    school_compact = normalize_compact(school)
    token_set = set(school_norm.split())

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        display_name = render_school_name_from_file(path, year)
        display_norm = normalize_for_match(display_name)
        display_compact = normalize_compact(display_name)
        score = 0

        if school_norm == display_norm:
            score += 1000
        if school_compact == display_compact:
            score += 900
        if school_norm in display_norm:
            score += 300
        if school_compact and school_compact in display_compact:
            score += 250

        display_tokens = set(display_norm.split())
        score += len(token_set & display_tokens) * 10

        scored.append((score, path))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    best_score, best_path = scored[0]
    if best_score <= 0:
        available = ", ".join(render_school_name_from_file(p, year) for p in candidates)
        raise RuntimeError(
            f"Could not match school '{school}' for year {year}. Available: {available}"
        )

    return best_path


def extract_degree_headings(pdf_path: Path, include_graduate: bool) -> list[DegreeHeading]:
    reader = PdfReader(str(pdf_path))
    headings: list[DegreeHeading] = []
    seen: set[tuple[str, str, str | None]] = set()

    excluded_prefixes = (
        "course",
        "courses",
        "curriculum",
        "sample",
        "fall",
        "spring",
        "summer",
        "grade",
        "gpa",
        "college of",
        "school of",
    )

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_lines = [normalize_text(raw_line) for raw_line in text.splitlines()]
        page_lines = [line for line in page_lines if line]

        for idx, line in enumerate(page_lines):
            candidates: list[str] = [line]

            # Some headings are split across two lines:
            #   Multidisciplinary Engineering/Acoustical Engineering
            #   Concentration, BSE
            if idx > 0 and SPLIT_HEADING_SUFFIX_RE.match(line):
                candidates.append(f"{page_lines[idx - 1]} {line}")

            for candidate_line in candidates:
                match = HEADING_RE.match(candidate_line)
                if not match:
                    continue

                title = normalize_text(match.group("title"))
                degree = normalize_text(match.group("degree"))
                variant_raw = match.group("variant")
                variant = normalize_text(variant_raw) if variant_raw else None

                if not include_graduate and not degree.startswith("B"):
                    continue

                title_norm = normalize_for_match(title)
                if not title_norm:
                    continue
                if any(title_norm.startswith(prefix) for prefix in excluded_prefixes):
                    continue

                # Require nearby section context so course-list lines that match
                # "..., BME" are excluded.
                lookahead = [
                    normalize_for_match(page_lines[j])
                    for j in range(idx + 1, min(len(page_lines), idx + 9))
                ]
                has_about = any(
                    line.startswith("about the ") and "program" in line for line in lookahead
                )
                has_degree_requirements = any("degree requirements" in line for line in lookahead)
                if not (has_about or has_degree_requirements):
                    continue

                # Extra filters to avoid accidental list/table lines.
                if "credits" in title_norm:
                    continue
                if COURSE_CODE_RE.search(title):
                    continue
                if title_norm in {"concentration", "track", "option"}:
                    continue

                key = (title, degree, variant)
                if key in seen:
                    continue
                seen.add(key)

                headings.append(
                    DegreeHeading(
                        title=title,
                        degree=degree,
                        variant=variant,
                        page=page_idx,
                        raw_heading=candidate_line,
                    )
                )

    headings.sort(key=lambda h: (h.page, h.title, h.degree, h.variant or ""))
    return headings


def default_output_paths(year: int, school_label: str) -> tuple[Path, Path]:
    stem = f"{year}_{slugify(school_label)}_majors"
    return OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.md"


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append(f"# {payload['school']} Majors ({payload['year']})")
    lines.append("")
    lines.append(f"- Source PDF: `{payload['source_pdf']}`")
    lines.append(f"- Total majors found: {payload['total_majors']}")
    lines.append("")

    majors = payload.get("majors", [])
    for i, major in enumerate(majors, start=1):
        suffix = f" ({major['variant']})" if major.get("variant") else ""
        lines.append(f"{i}. {major['title']}, {major['degree']}{suffix} (page {major['page']})")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.list_schools:
        schools = find_year_combined_pdfs(args.year)
        if not schools:
            print(f"No combined school PDFs found for {args.year} in {CATALOG_DIR}")
            return 0

        for path in schools:
            print(render_school_name_from_file(path, args.year))
        return 0

    if args.pdf is None and not args.school:
        print("[error] Provide either --pdf or --school", file=sys.stderr)
        return 2

    if args.pdf is not None:
        pdf_path = args.pdf
        school_label = args.school or pdf_path.stem
    else:
        try:
            pdf_path = choose_pdf_for_school(args.year, args.school)
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        school_label = args.school

    if not pdf_path.exists():
        print(f"[error] PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    majors = extract_degree_headings(pdf_path, include_graduate=args.include_graduate)

    json_out, md_out = default_output_paths(args.year, school_label)
    if args.json_out:
        json_out = args.json_out
    if args.md_out:
        md_out = args.md_out

    payload = {
        "year": args.year,
        "school": school_label,
        "source_pdf": str(pdf_path),
        "total_majors": len(majors),
        "include_graduate": args.include_graduate,
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

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_out, payload)

    print(f"School: {school_label}")
    print(f"Source PDF: {pdf_path}")
    print(f"Majors found: {len(majors)}")
    print(f"Wrote JSON: {json_out}")
    print(f"Wrote Markdown: {md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
