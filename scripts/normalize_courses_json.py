#!/usr/bin/env python3
"""Normalize parsed catalog courses JSON for relational loading.

Input:
  - data/courses.json

Outputs:
  - data/normalized/courses_normalized.json
  - data/normalized/prerequisite_edges.json
  - data/normalized/normalization_issues.json
  - data/normalized/normalization_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FULL_CODE_RE = re.compile(r"\b([A-Z]{2,6})\s*[- ]?\s*([0-9]{3,6}[A-Z]?)\b")
CONTINUATION_NUMBER_RE = re.compile(
    r"(?:,|/|;|\bor\b|\band\b)\s*([0-9]{3,6}[A-Z]?)\b",
    re.IGNORECASE,
)


def compact(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed or None


def normalize_course_code(subject: str, number: str) -> str:
    return f"{subject.upper()} {number.upper()}"


def normalize_catalog_number(number: str) -> str | None:
    token = number.strip().upper()
    match = re.match(r"^([0-9]{3,6})([A-Z]?)$", token)
    if not match:
        return None

    digits, suffix = match.groups()

    # Purdue catalog often uses 5-digit catalog numbers; in narrative text
    # prerequisites frequently appear as 3-digit shorthand (e.g., "CS 251").
    if len(digits) == 3:
        digits = f"{digits}00"
    elif len(digits) == 4:
        digits = f"{digits}0"
    elif len(digits) == 5:
        pass
    elif len(digits) == 6 and digits.endswith("0"):
        # OCR occasionally appends a trailing zero (e.g., 483000 -> 48300).
        digits = digits[:-1]
    else:
        return None

    return f"{digits}{suffix}"


def infer_prerequisite_logic(prerequisite_text: str | None) -> str | None:
    if not prerequisite_text:
        return None

    text = prerequisite_text.lower()
    has_or = bool(re.search(r"\bor\b|/", text))
    has_and = bool(re.search(r"\band\b|,|;", text))

    if has_or and has_and:
        return "mixed"
    if has_or:
        return "or"
    if has_and:
        return "and"
    return "unspecified"


def extract_prereq_codes(prerequisite_text: str | None) -> list[str]:
    if not prerequisite_text:
        return []

    text = compact(prerequisite_text) or ""
    codes: list[str] = []
    seen: set[str] = set()

    full_matches = list(FULL_CODE_RE.finditer(text))
    for match in full_matches:
        normalized_number = normalize_catalog_number(match.group(2))
        if normalized_number is None:
            continue
        code = normalize_course_code(match.group(1), normalized_number)
        if code not in seen:
            seen.add(code)
            codes.append(code)

    # Handle shorthand references like "MA 16010, 16500, or 16600."
    for bare in CONTINUATION_NUMBER_RE.finditer(text):
        normalized_number = normalize_catalog_number(bare.group(1))
        if normalized_number is None:
            continue
        prior = None
        for match in reversed(full_matches):
            if match.start() < bare.start():
                prior = match
                break
        if prior is None:
            continue

        subject = prior.group(1).upper()
        code = normalize_course_code(subject, normalized_number)
        if code not in seen:
            seen.add(code)
            codes.append(code)

    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize parsed courses JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/courses.json"),
        help="Path to parsed courses.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/normalized"),
        help="Directory to write normalized outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_courses: list[dict[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))

    normalized_courses: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for raw in raw_courses:
        subject = compact(raw.get("subject")) or ""
        number = compact(raw.get("number")) or ""
        course_code = normalize_course_code(subject, number)

        prerequisite_text = compact(raw.get("prerequisite_text"))
        prerequisite_codes = extract_prereq_codes(prerequisite_text)
        prerequisite_logic = infer_prerequisite_logic(prerequisite_text)

        normalized = {
            "course_code": course_code,
            "subject": subject.upper(),
            "catalog_number": number.upper(),
            "title": compact(raw.get("title")) or "",
            "credit_hours": raw.get("credit_hours"),
            "description": compact(raw.get("description")),
            "prerequisite_text": prerequisite_text,
            "prerequisite_logic": prerequisite_logic,
            "prerequisite_codes": prerequisite_codes,
            "source_page_start": raw.get("source_page_start"),
            "source_page_end": raw.get("source_page_end"),
        }
        normalized_courses.append(normalized)

        if prerequisite_text and not prerequisite_codes:
            issues.append(
                {
                    "course_code": course_code,
                    "issue_type": "no_course_codes_in_prerequisite_text",
                    "details": prerequisite_text,
                }
            )

        for idx, prereq_code in enumerate(prerequisite_codes, start=1):
            edges.append(
                {
                    "course_code": course_code,
                    "prerequisite_code": prereq_code,
                    "sequence": idx,
                    "logic_hint": prerequisite_logic,
                    "source_text": prerequisite_text,
                }
            )

    known_codes = {course["course_code"] for course in normalized_courses}
    for edge in edges:
        if edge["prerequisite_code"] not in known_codes:
            issues.append(
                {
                    "course_code": edge["course_code"],
                    "issue_type": "prerequisite_course_not_in_catalog",
                    "details": edge["prerequisite_code"],
                }
            )

    summary = {
        "source_file": str(args.input),
        "courses_total": len(normalized_courses),
        "courses_with_prerequisite_text": sum(
            1 for course in normalized_courses if course["prerequisite_text"]
        ),
        "courses_with_prerequisite_codes": sum(
            1 for course in normalized_courses if course["prerequisite_codes"]
        ),
        "prerequisite_edges_total": len(edges),
        "issues_total": len(issues),
    }

    (args.output_dir / "courses_normalized.json").write_text(
        json.dumps(normalized_courses, indent=2), encoding="utf-8"
    )
    (args.output_dir / "prerequisite_edges.json").write_text(
        json.dumps(edges, indent=2), encoding="utf-8"
    )
    (args.output_dir / "normalization_issues.json").write_text(
        json.dumps(issues, indent=2), encoding="utf-8"
    )
    (args.output_dir / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Normalization complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
