#!/usr/bin/env python3
"""Extract degrees and degree requirements from Purdue catalog PDF.

This script performs a best-effort structured extraction from pages that contain
"Degree Requirements", then links requirement lines to course codes.

Outputs (JSON):
  - degrees.json
  - degree_requirements.json
  - degree_requirement_courses.json
  - degree_extraction_issues.json
  - degree_extraction_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


AWARD_PATTERN = r"(?:AAE|AS|AAS|BA|BFA|BM|BS|BSE|BSN|DNP|DVM|MA|MBA|MS|MSW|PharmD|PhD|PHD)"
DEGREE_LINE_RE = re.compile(
    rf"^[A-Za-z0-9&/'().,:\- ]+,\s*{AWARD_PATTERN}\s*$",
    re.IGNORECASE,
)
DEGREE_IN_TEXT_RE = re.compile(
    rf"([A-Za-z0-9&/'().,:\- ]+,\s*{AWARD_PATTERN})",
    re.IGNORECASE,
)
COURSE_LINE_RE = re.compile(
    r"^(?P<course_code>[A-Z]{2,6}\s+\d{5}[A-Z]?)\s*-\s*(?P<title>.+)$"
)
REQ_GROUP_INLINE_RE = re.compile(
    r"^(?P<name>.+?)\s*\((?P<credits>[0-9]+(?:\.[0-9]+)?(?:\s*[-–]\s*[0-9]+(?:\.[0-9]+)?)?)\s*credits\)\s*$",
    re.IGNORECASE,
)
REQ_CREDITS_ONLY_RE = re.compile(
    r"^\((?P<credits>[0-9]+(?:\.[0-9]+)?(?:\s*[-–]\s*[0-9]+(?:\.[0-9]+)?)?)\s*credits\)\s*$",
    re.IGNORECASE,
)
TITLE_STOP_RE = re.compile(
    r"^(?:\d+(?:-\d+)?\s+Credits|Fall\s+\d|Spring\s+\d|Summer\s+\d|Credits?:|[A-Z]{2,6}\s+\d{5}[A-Z]?|About the Program|Degree Requirements)$",
    re.IGNORECASE,
)


REQ_KEYWORDS = (
    "course",
    "courses",
    "requirement",
    "requirements",
    "elective",
    "electives",
    "major",
    "minor",
    "core",
    "concentration",
)


def normalize_line(line: str) -> str:
    cleaned = line.replace("\u00a0", " ").replace("\uf0b7", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("•")
    return cleaned


def normalize_name(name: str) -> str:
    name = normalize_line(name)
    name = re.sub(r"^Notes\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^and\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_award(degree_name: str) -> str | None:
    match = re.search(rf",\s*({AWARD_PATTERN})\s*$", degree_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def infer_level(award: str | None) -> str | None:
    if award is None:
        return None
    award_u = award.upper()
    if award_u in {"BS", "BA", "BFA", "BM", "BSE", "BSN", "AS", "AAS"}:
        return "undergraduate"
    if award_u in {"MS", "MA", "MBA", "MSW", "PHD", "PHD", "DNP", "PHARMD", "DVM"}:
        return "graduate_or_professional"
    return "other"


def infer_confidence(name: str) -> str:
    lowered = name.lower()
    if name.startswith("UNKNOWN_PAGE_"):
        return "low"
    if lowered in {"concentration, bs", "studies concentration, bs"}:
        return "low"
    if lowered.startswith("and "):
        return "low"
    if len(name.split()) <= 2:
        return "low"
    if ":" in name or "concentration" in lowered:
        return "high"
    return "medium"


def looks_like_requirement_name(line: str) -> bool:
    lower = line.lower()
    return any(keyword in lower for keyword in REQ_KEYWORDS)


def infer_degree_name(
    lines: list[str],
    about_idx: int,
    degree_lines_on_page: list[str],
    recent_heading: tuple[str, int] | None,
    page_num: int,
) -> tuple[str, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []

    # 1) Try immediate lines above "About the Program"
    if about_idx != -1:
        title_parts: list[str] = []
        for idx in range(about_idx - 1, max(-1, about_idx - 5), -1):
            line = lines[idx]
            if TITLE_STOP_RE.match(line):
                break
            title_parts.append(line)
        title_parts.reverse()
        if title_parts:
            candidate = normalize_name(" ".join(title_parts))
            match = DEGREE_IN_TEXT_RE.search(candidate)
            if match:
                return normalize_name(match.group(1)), issues
            if DEGREE_LINE_RE.match(candidate):
                return normalize_name(candidate), issues

    # 2) Try any explicit degree line on page
    if degree_lines_on_page:
        return normalize_name(degree_lines_on_page[0]), issues

    # 3) Use recent heading if nearby
    if recent_heading and page_num - recent_heading[1] <= 2:
        issues.append(
            {
                "issue_type": "degree_name_inferred_from_recent_heading",
                "details": f"Borrowed heading from page {recent_heading[1]}",
            }
        )
        return normalize_name(recent_heading[0]), issues

    # 4) Last resort
    issues.append(
        {
            "issue_type": "degree_name_not_found",
            "details": "Could not confidently identify degree title on or near page.",
        }
    )
    return f"UNKNOWN_PAGE_{page_num}", issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract degree requirements from PDF.")
    parser.add_argument(
        "--input-pdf",
        type=Path,
        default=Path("2025-26+University+Catalog-Final.pdf"),
        help="Path to catalog PDF",
    )
    parser.add_argument(
        "--courses-json",
        type=Path,
        default=Path("data/courses.json"),
        help="Path to deduped courses JSON for course-code validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/degree_extracted"),
        help="Directory where degree extraction JSON files are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog_courses = json.loads(args.courses_json.read_text(encoding="utf-8"))
    known_course_codes = {course["code"] for course in catalog_courses}

    reader = PdfReader(str(args.input_pdf))

    degrees_by_name: dict[str, dict[str, Any]] = {}
    requirements_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    req_courses_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []

    recent_heading: tuple[str, int] | None = None
    pages_scanned = 0
    requirement_pages = 0

    for page_num, page in enumerate(reader.pages, start=1):
        pages_scanned += 1
        text = page.extract_text() or ""
        lines = [normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        degree_lines_on_page = [line for line in lines if DEGREE_LINE_RE.match(line)]
        if degree_lines_on_page:
            recent_heading = (degree_lines_on_page[-1], page_num)

        about_idx = next((i for i, line in enumerate(lines) if line == "About the Program"), -1)
        degree_req_idx = next((i for i, line in enumerate(lines) if "Degree Requirements" in line), -1)

        if degree_req_idx == -1:
            continue

        # We only keep likely degree pages.
        req_group_indexes: list[tuple[int, str, str]] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            inline = REQ_GROUP_INLINE_RE.match(line)
            if inline and looks_like_requirement_name(inline.group("name")):
                req_group_indexes.append((idx, normalize_name(inline.group("name")), inline.group("credits")))
                idx += 1
                continue

            # Handle two-line requirement groups:
            # "Departmental/Program Major Courses" + "(54 credits)"
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                credits_only = REQ_CREDITS_ONLY_RE.match(next_line)
                if credits_only and looks_like_requirement_name(line):
                    req_group_indexes.append((idx, normalize_name(line), credits_only.group("credits")))
                    idx += 2
                    continue
            idx += 1

        course_lines: list[tuple[int, str, str]] = []
        for idx, line in enumerate(lines):
            match = COURSE_LINE_RE.match(line)
            if not match:
                continue
            code = normalize_line(match.group("course_code")).upper()
            title = normalize_line(match.group("title"))
            course_lines.append((idx, code, title))

        likely_degree_page = bool(req_group_indexes) and bool(course_lines) and (
            about_idx != -1 or degree_req_idx <= 12
        )
        if not likely_degree_page:
            continue

        requirement_pages += 1

        degree_name, inferred_issues = infer_degree_name(
            lines=lines,
            about_idx=about_idx,
            degree_lines_on_page=degree_lines_on_page,
            recent_heading=recent_heading,
            page_num=page_num,
        )

        for issue in inferred_issues:
            issues.append({"page": page_num, "degree_name": degree_name, **issue})

        degree_name = normalize_name(degree_name)
        award = extract_award(degree_name)
        level = infer_level(award)
        confidence = infer_confidence(degree_name)

        if degree_name not in degrees_by_name:
            degrees_by_name[degree_name] = {
                "degree_name": degree_name,
                "award": award,
                "level": level,
                "confidence": confidence,
                "source_page_start": page_num,
                "source_page_end": page_num,
                "source_pages": [page_num],
                "requirement_pages_count": 1,
            }
        else:
            degree = degrees_by_name[degree_name]
            degree["source_page_end"] = max(degree["source_page_end"], page_num)
            if page_num not in degree["source_pages"]:
                degree["source_pages"].append(page_num)
            degree["requirement_pages_count"] += 1

        if confidence == "low":
            issues.append(
                {
                    "page": page_num,
                    "degree_name": degree_name,
                    "issue_type": "low_confidence_degree_name",
                    "details": "Degree title appears truncated or generic.",
                }
            )

        # Register requirement groups found on this page.
        page_req_keys: list[tuple[int, tuple[str, str, str]]] = []
        for line_idx, req_name, credits_text in req_group_indexes:
            req_key = (degree_name, req_name, credits_text)
            if req_key not in requirements_by_key:
                requirements_by_key[req_key] = {
                    "degree_name": degree_name,
                    "requirement_name": req_name,
                    "credits_text": credits_text,
                    "source_page_start": page_num,
                    "source_page_end": page_num,
                    "source_pages": [page_num],
                }
            else:
                req = requirements_by_key[req_key]
                req["source_page_end"] = max(req["source_page_end"], page_num)
                if page_num not in req["source_pages"]:
                    req["source_pages"].append(page_num)
            page_req_keys.append((line_idx, req_key))

        page_req_keys.sort(key=lambda item: item[0])

        # Link course lines to closest preceding requirement group on same page.
        for course_idx, course_code, course_title in course_lines:
            assigned_req_key = None
            for line_idx, req_key in reversed(page_req_keys):
                if line_idx <= course_idx:
                    assigned_req_key = req_key
                    break

            req_name = assigned_req_key[1] if assigned_req_key else "UNASSIGNED_ON_PAGE"
            credits_text = assigned_req_key[2] if assigned_req_key else ""
            link_key = (degree_name, req_name, course_code)
            if link_key not in req_courses_by_key:
                req_courses_by_key[link_key] = {
                    "degree_name": degree_name,
                    "requirement_name": req_name,
                    "credits_text": credits_text,
                    "course_code": course_code,
                    "course_title_on_page": course_title,
                    "course_in_catalog": course_code in known_course_codes,
                    "source_page_start": page_num,
                    "source_page_end": page_num,
                    "source_pages": [page_num],
                }
            else:
                link = req_courses_by_key[link_key]
                link["source_page_end"] = max(link["source_page_end"], page_num)
                if page_num not in link["source_pages"]:
                    link["source_pages"].append(page_num)

    degrees = sorted(degrees_by_name.values(), key=lambda d: d["degree_name"])
    requirements = sorted(
        requirements_by_key.values(),
        key=lambda r: (r["degree_name"], r["requirement_name"], r["credits_text"]),
    )
    requirement_courses = sorted(
        req_courses_by_key.values(),
        key=lambda rc: (rc["degree_name"], rc["requirement_name"], rc["course_code"]),
    )

    summary = {
        "source_pdf": str(args.input_pdf),
        "pages_scanned": pages_scanned,
        "requirement_pages_parsed": requirement_pages,
        "degrees_extracted": len(degrees),
        "requirements_extracted": len(requirements),
        "requirement_course_links_extracted": len(requirement_courses),
        "low_confidence_degrees": sum(1 for d in degrees if d["confidence"] == "low"),
        "issues_logged": len(issues),
    }

    (args.output_dir / "degrees.json").write_text(json.dumps(degrees, indent=2), encoding="utf-8")
    (args.output_dir / "degree_requirements.json").write_text(
        json.dumps(requirements, indent=2), encoding="utf-8"
    )
    (args.output_dir / "degree_requirement_courses.json").write_text(
        json.dumps(requirement_courses, indent=2), encoding="utf-8"
    )
    (args.output_dir / "degree_extraction_issues.json").write_text(
        json.dumps(issues, indent=2), encoding="utf-8"
    )
    (args.output_dir / "degree_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Degree extraction complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

