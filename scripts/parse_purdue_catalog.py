#!/usr/bin/env python3
"""Parse Purdue catalog PDF into structured JSON files.

Usage:
  ./.venv/bin/python scripts/parse_purdue_catalog.py \
    --input 2025-26+University+Catalog-Final.pdf \
    --output-dir backend/app/data/purdue_2025_26
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


COURSE_START_RE = re.compile(
    r"^(?P<subject>[A-Z]{2,6})\s+(?P<number>\d{5}[A-Z]?)\s*-\s*(?P<title>.+)$"
)
CREDIT_HOURS_RE = re.compile(r"Credit Hours:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PREREQ_RE = re.compile(
    r"Prerequisite[s]?:\s*(.*?)(?=(?:Corequisite|Learning Outcomes|Credits?:|\Z))",
    re.IGNORECASE | re.DOTALL,
)
REQ_LINE_RE = re.compile(r"\b(Degree Requirements?|Program Requirements?)\b", re.IGNORECASE)
NOISE_LINE_RE = re.compile(
    r"^(Table of Contents|College Department Course Subject Code Level Description)$",
    re.IGNORECASE,
)


@dataclass
class CourseBlock:
    subject: str
    number: str
    title: str
    start_page: int
    end_page: int
    lines: list[str] = field(default_factory=list)

    @property
    def code(self) -> str:
        return f"{self.subject} {self.number}"

    def to_record(self) -> dict:
        title, body_lines = self._extract_title_and_body()
        text = " ".join(body_lines)
        compact_text = compact_whitespace(text)
        credit_hours = extract_credit_hours(compact_text)
        prereqs = extract_prereqs(compact_text)
        description = extract_description(compact_text)

        return {
            "code": self.code,
            "subject": self.subject,
            "number": self.number,
            "title": title,
            "credit_hours": credit_hours,
            "prerequisite_text": prereqs,
            "description": description,
            "source_page_start": self.start_page,
            "source_page_end": self.end_page,
            "raw_text": compact_text,
        }

    def _extract_title_and_body(self) -> tuple[str, list[str]]:
        marker_re = re.compile(
            r"^(Credit Hours:|Prerequisite[s]?:|Corequisite[s]?:|Learning Outcomes:|Credits?:)",
            re.IGNORECASE,
        )

        title_parts = [self.title]
        consumed = 0
        max_wrap_lines = 2

        for line in self.lines:
            if consumed >= max_wrap_lines:
                break
            if marker_re.match(line):
                break
            if len(line.split()) > 9:
                break

            title_parts.append(line)
            consumed += 1

        title = compact_whitespace(" ".join(title_parts))
        return title, self.lines[consumed:]


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\uf0b7", " ").strip()
    return compact_whitespace(line)


def extract_credit_hours(text: str) -> float | None:
    match = CREDIT_HOURS_RE.search(text)
    if not match:
        return None
    return float(match.group(1))


def extract_prereqs(text: str) -> str | None:
    match = PREREQ_RE.search(text)
    if not match:
        return None
    prereq_text = compact_whitespace(match.group(1))
    return prereq_text if prereq_text else None


def extract_description(text: str) -> str:
    # Keep description to first major metadata marker for readability.
    markers = ["Learning Outcomes:", "Prerequisite:", "Prerequisites:", "Credits:"]
    end_idx = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = min(end_idx, idx)

    trimmed = text[:end_idx]
    # Remove leading "Credit Hours: X.00." block if present.
    trimmed = re.sub(r"^Credit Hours:\s*[0-9]+(?:\.[0-9]+)?\.\s*", "", trimmed, flags=re.I)
    return compact_whitespace(trimmed)


def iter_pages(reader: PdfReader) -> Iterable[tuple[int, list[str]]]:
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = []
        for raw in text.splitlines():
            line = normalize_line(raw)
            if not line:
                continue
            if NOISE_LINE_RE.match(line):
                continue
            lines.append(line)
        yield idx, lines


def parse_course_blocks(reader: PdfReader) -> tuple[list[dict], dict]:
    blocks: list[CourseBlock] = []
    active: CourseBlock | None = None
    pages_scanned = 0
    pages_with_courses = 0

    for page_num, lines in iter_pages(reader):
        pages_scanned += 1
        page_has_course = False

        for line in lines:
            match = COURSE_START_RE.match(line)
            if match:
                page_has_course = True
                if active is not None:
                    blocks.append(active)
                active = CourseBlock(
                    subject=match.group("subject"),
                    number=match.group("number"),
                    title=match.group("title"),
                    start_page=page_num,
                    end_page=page_num,
                    lines=[],
                )
                # Keep the line after title stripped out to avoid duplication in description.
                continue

            if active is not None:
                active.end_page = page_num
                active.lines.append(line)

        if page_has_course:
            pages_with_courses += 1

    if active is not None:
        blocks.append(active)

    records = [block.to_record() for block in blocks]

    diagnostics = {
        "pages_scanned": pages_scanned,
        "pages_with_course_starts": pages_with_courses,
        "course_blocks_detected": len(blocks),
    }
    return records, diagnostics


def dedupe_courses(courses: list[dict]) -> tuple[list[dict], list[dict]]:
    by_code: dict[str, list[dict]] = {}
    for course in courses:
        by_code.setdefault(course["code"], []).append(course)

    deduped: list[dict] = []
    duplicates: list[dict] = []

    def score(record: dict) -> tuple[int, int, int]:
        return (
            1 if record.get("prerequisite_text") else 0,
            1 if record.get("credit_hours") is not None else 0,
            len(record.get("description") or ""),
        )

    for code, records in by_code.items():
        best = max(records, key=score)
        deduped.append(best)
        if len(records) > 1:
            duplicates.append(
                {
                    "code": code,
                    "occurrences": len(records),
                    "selected_source_page_start": best["source_page_start"],
                    "selected_source_page_end": best["source_page_end"],
                }
            )

    deduped.sort(key=lambda c: c["code"])
    duplicates.sort(key=lambda d: d["occurrences"], reverse=True)
    return deduped, duplicates


def parse_requirement_snippets(reader: PdfReader, window: int = 4) -> list[dict]:
    snippets: list[dict] = []
    for page_num, lines in iter_pages(reader):
        joined = "\n".join(lines)
        if not REQ_LINE_RE.search(joined):
            continue

        sample_lines = lines[: 30 + window]
        snippets.append(
            {
                "page": page_num,
                "matched_line_count": len([line for line in lines if REQ_LINE_RE.search(line)]),
                "sample_text": compact_whitespace(" ".join(sample_lines))[:3000],
            }
        )
    return snippets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Purdue catalog PDF into JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Path to source PDF")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where JSON outputs are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(args.input))

    courses_raw, diagnostics = parse_course_blocks(reader)
    courses_deduped, duplicate_summary = dedupe_courses(courses_raw)
    requirement_snippets = parse_requirement_snippets(reader)

    courses_path = args.output_dir / "courses.json"
    courses_raw_path = args.output_dir / "courses_raw.json"
    duplicates_path = args.output_dir / "course_duplicates.json"
    snippets_path = args.output_dir / "degree_requirement_snippets.json"
    diagnostics_path = args.output_dir / "parse_diagnostics.json"

    courses_path.write_text(json.dumps(courses_deduped, indent=2), encoding="utf-8")
    courses_raw_path.write_text(json.dumps(courses_raw, indent=2), encoding="utf-8")
    duplicates_path.write_text(json.dumps(duplicate_summary, indent=2), encoding="utf-8")
    snippets_path.write_text(json.dumps(requirement_snippets, indent=2), encoding="utf-8")
    diagnostics_path.write_text(
        json.dumps(
            {
                "source_pdf": str(args.input),
                "total_pdf_pages": len(reader.pages),
                "course_extraction": diagnostics,
                "unique_courses_after_dedupe": len(courses_deduped),
                "course_codes_with_duplicates": len(duplicate_summary),
                "degree_requirement_pages_found": len(requirement_snippets),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {courses_path}")
    print(f"Wrote {courses_raw_path}")
    print(f"Wrote {duplicates_path}")
    print(f"Wrote {snippets_path}")
    print(f"Wrote {diagnostics_path}")
    print(f"Raw course blocks extracted: {len(courses_raw)}")
    print(f"Unique course codes: {len(courses_deduped)}")


if __name__ == "__main__":
    main()
