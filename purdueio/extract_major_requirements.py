#!/usr/bin/env python3
"""Extract requirement rules for a chosen major from College of Science 2024 PDF.

Examples:
  python purdueio/extract_cs_machine_intelligence_requirements.py --major "Data Science"
  python purdueio/extract_cs_machine_intelligence_requirements.py --major "Data Science" --variant "STAT"
  python purdueio/extract_cs_machine_intelligence_requirements.py --major "Computer Science" --concentration "Machine Intelligence"
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


DEFAULT_PDF = Path("purdueio/data/catalog_pdfs/2024-CollegeofScience-Combined.pdf")
OUT_DIR = Path("purdueio/data/extracted")

HEADING_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9&'()/:,\.\- ]+?),\s*(?P<degree>[A-Z/]{2,8})(?:\s*\((?P<variant>[^)]+)\))?$"
)
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s*\d{3,5}\b")
SPLIT_HEADING_SUFFIX_RE = re.compile(
    r"^(Concentration|Track|Option),\s*[A-Z/]{2,8}(?:\s*\([^)]+\))?$",
    re.I,
)
COURSE_RE = re.compile(
    r"•\s*"
    r"(?P<subject>[A-Z]{2,5})\s*(?P<number>\d{5})"
    r"\s*-\s*"
    r"(?P<title>.*?)"
    r"\s+Credits:\s*(?P<credits>[0-9]+(?:\.[0-9]+)?)"
)


@dataclass
class LineRecord:
    idx: int
    page: int
    text: str


@dataclass
class HeadingMatch:
    line_index: int
    page: int
    line_text: str
    title: str
    degree: str
    variant: str | None
    score: int


@dataclass
class CourseRow:
    course_code: str
    title: str
    credits: float
    raw_line: str


@dataclass
class ParsedCourse:
    course: CourseRow
    connector: str | None  # 'or', 'and', or None


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = " ".join(value.split())
    return value.strip()


def normalize_for_match(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return " ".join(value.split())


def slugify(value: str) -> str:
    text = normalize_for_match(value)
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "major"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract degree requirement rules for a selected major from 2024 College of Science PDF."
    )
    parser.add_argument("--major", required=True, help="Major name to match, e.g. 'Data Science'.")
    parser.add_argument(
        "--degree",
        default="BS",
        help="Degree suffix in heading (default: BS).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional variant in heading parenthesis, e.g. CS or STAT for Data Science.",
    )
    parser.add_argument(
        "--concentration",
        default=None,
        help="Optional concentration keyword in heading text, e.g. Machine Intelligence.",
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to combined college PDF.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional output markdown path.")
    parser.add_argument(
        "--list-headings",
        action="store_true",
        help="List detected degree headings and exit.",
    )
    return parser.parse_args()


def extract_lines(pdf_path: Path) -> list[LineRecord]:
    reader = PdfReader(str(pdf_path))
    lines: list[LineRecord] = []
    idx = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            clean = normalize_text(raw_line)
            if not clean:
                continue
            lines.append(LineRecord(idx=idx, page=page_num, text=clean))
            idx += 1
    return lines


def find_degree_headings(lines: list[LineRecord]) -> list[HeadingMatch]:
    headings: list[HeadingMatch] = []
    seen: set[tuple[int, str, str, str | None]] = set()

    page_map: dict[int, list[LineRecord]] = {}
    for rec in lines:
        page_map.setdefault(rec.page, []).append(rec)

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

    for page, page_records in page_map.items():
        for i, rec in enumerate(page_records):
            candidates: list[tuple[int, str]] = [(rec.idx, rec.text)]
            if i > 0 and SPLIT_HEADING_SUFFIX_RE.match(rec.text):
                merged = f"{page_records[i - 1].text} {rec.text}"
                candidates.append((page_records[i - 1].idx, merged))

            for line_index, candidate_line in candidates:
                m = HEADING_RE.match(candidate_line)
                if not m:
                    continue

                title = m.group("title")
                degree = m.group("degree")
                variant = m.group("variant")
                title_norm = normalize_for_match(title)

                if not title_norm:
                    continue
                if any(title_norm.startswith(prefix) for prefix in excluded_prefixes):
                    continue
                if "credits" in title_norm:
                    continue
                if COURSE_CODE_RE.search(title):
                    continue
                if title_norm in {"concentration", "track", "option"}:
                    continue

                lookahead = [
                    normalize_for_match(page_records[j].text)
                    for j in range(i + 1, min(len(page_records), i + 9))
                ]
                has_about = any(
                    line.startswith("about the ") and "program" in line for line in lookahead
                )
                has_degree_requirements = any("degree requirements" in line for line in lookahead)
                if not (has_about or has_degree_requirements):
                    continue

                key = (line_index, title, degree, variant)
                if key in seen:
                    continue
                seen.add(key)

                headings.append(
                    HeadingMatch(
                        line_index=line_index,
                        page=page,
                        line_text=candidate_line,
                        title=title,
                        degree=degree,
                        variant=variant,
                        score=0,
                    )
                )
    headings.sort(key=lambda h: h.line_index)
    return headings


def choose_heading(
    headings: list[HeadingMatch],
    major: str,
    degree: str,
    variant: str | None,
    concentration: str | None,
) -> tuple[HeadingMatch, list[HeadingMatch]]:
    major_norm = normalize_for_match(major)
    degree_norm = degree.upper()
    variant_norm = normalize_for_match(variant) if variant else None
    concentration_norm = normalize_for_match(concentration) if concentration else None

    candidates: list[HeadingMatch] = []
    for h in headings:
        if h.degree.upper() != degree_norm:
            continue

        title_norm = normalize_for_match(h.title)
        full_norm = normalize_for_match(h.line_text)

        if major_norm not in title_norm and major_norm not in full_norm:
            continue

        if variant_norm:
            heading_variant = normalize_for_match(h.variant or "")
            if variant_norm != heading_variant:
                continue

        score = 0
        if title_norm == major_norm:
            score += 120
        if title_norm.startswith(major_norm):
            score += 70
        if major_norm in title_norm:
            score += 40

        if concentration_norm and concentration_norm in full_norm:
            score += 90

        if "honors" in title_norm and "honors" not in major_norm:
            score -= 15

        h.score = score
        candidates.append(h)

    if not candidates:
        raise RuntimeError(
            f"No matching heading found for major='{major}', degree='{degree}'"
            + (f", variant='{variant}'" if variant else "")
            + (f", concentration='{concentration}'" if concentration else "")
        )

    candidates.sort(key=lambda x: (-x.score, x.page, x.line_index))
    return candidates[0], candidates


def find_section_end_line(
    lines: list[LineRecord],
    selected: HeadingMatch,
    headings: list[HeadingMatch],
) -> int:
    next_heading_idx = len(lines)
    for h in headings:
        if h.line_index > selected.line_index:
            next_heading_idx = h.line_index
            break

    stop_markers = (
        "minor",
        "pre-program",
    )
    cutoff = next_heading_idx
    for rec in lines:
        if rec.idx <= selected.line_index:
            continue
        if rec.idx >= cutoff:
            break

        low = rec.text.lower()
        if low in stop_markers:
            cutoff = rec.idx
            break
        if low.startswith("department of "):
            cutoff = rec.idx
            break

    return cutoff


def parse_choose_count(block_lines: list[str]) -> int | None:
    for line in block_lines:
        low = line.lower()
        if "choose one" in low:
            return 1
        if "choose two" in low:
            return 2
        m = re.search(r"choose\s+(\d+)", low)
        if m:
            return int(m.group(1))
    return None


def parse_courses_with_connectors(block_lines: list[str]) -> list[ParsedCourse]:
    rows: list[ParsedCourse] = []
    seen: set[str] = set()

    for line in block_lines:
        m = COURSE_RE.search(line)
        if not m:
            continue

        key = f"{m.group('subject')}{m.group('number')}|{m.group('title')}|{m.group('credits')}"
        if key in seen:
            continue
        seen.add(key)

        trailing = normalize_text(line[m.end() :]).lower()
        connector: str | None = None
        if trailing.startswith("or"):
            connector = "or"
        elif trailing.startswith("and"):
            connector = "and"

        rows.append(
            ParsedCourse(
                course=CourseRow(
                    course_code=f"{m.group('subject')}{m.group('number')}",
                    title=normalize_text(m.group("title")),
                    credits=float(m.group("credits")),
                    raw_line=line,
                ),
                connector=connector,
            )
        )

    return rows


def build_options(parsed: list[ParsedCourse]) -> list[list[CourseRow]]:
    options: list[list[CourseRow]] = []
    current: list[CourseRow] = []

    for item in parsed:
        current.append(item.course)
        if item.connector == "and":
            continue

        options.append(current)
        current = []

    if current:
        options.append(current)

    return options


def split_connector_clusters(parsed: list[ParsedCourse]) -> list[list[ParsedCourse]]:
    if not parsed:
        return []

    clusters: list[list[ParsedCourse]] = []
    current: list[ParsedCourse] = []

    for i, item in enumerate(parsed):
        current.append(item)
        keep_link = item.connector in {"or", "and"}
        if keep_link:
            continue

        clusters.append(current)
        current = []

    if current:
        clusters.append(current)

    return clusters


def course_to_dict(course: CourseRow) -> dict[str, object]:
    return {
        "course_code": course.course_code,
        "title": course.title,
        "credits": course.credits,
        "raw_line": course.raw_line,
    }


def options_to_dict(options: list[list[CourseRow]]) -> list[dict[str, object]]:
    return [{"courses": [course_to_dict(c) for c in option]} for option in options]


def build_rules_for_block(block_lines: list[str]) -> tuple[list[dict[str, object]], list[CourseRow], int | None]:
    choose_count = parse_choose_count(block_lines)
    parsed = parse_courses_with_connectors(block_lines)
    courses = [p.course for p in parsed]

    if not parsed:
        return [], [], choose_count

    rules: list[dict[str, object]] = []

    if choose_count is not None:
        options = build_options(parsed)
        rules.append(
            {
                "type": "choice_group",
                "choose": choose_count,
                "options": options_to_dict(options),
            }
        )
        return rules, courses, choose_count

    clusters = split_connector_clusters(parsed)
    for cluster in clusters:
        options = build_options(cluster)
        if len(options) == 1 and len(options[0]) == 1:
            rules.append({"type": "course", "course": course_to_dict(options[0][0])})
            continue
        if len(options) == 1 and len(options[0]) > 1:
            rules.append({"type": "all_of", "courses": [course_to_dict(c) for c in options[0]]})
            continue

        rules.append(
            {
                "type": "choice_group",
                "choose": 1,
                "options": options_to_dict(options),
            }
        )

    return rules, courses, choose_count


def extract_blocks(section_lines: list[LineRecord]) -> list[dict[str, object]]:
    plain_lines = [rec.text for rec in section_lines]

    # Prefer the dedicated section heading "Degree Requirements" over broader
    # narrative lines like "Curriculum and Degree Requirements for ...".
    start = None
    for i, line in enumerate(plain_lines):
        if normalize_for_match(line) == "degree requirements":
            start = i
            break

    if start is None:
        for i, line in enumerate(plain_lines):
            if "degree requirements" in line.lower():
                start = i
                break

    if start is None:
        start = 0

    # Some majors include narrative + Science Core text before major-specific
    # course blocks. Shift start forward when we detect common major anchors.
    major_anchors = (
        "major courses",
        "departmental program major courses",
        "required major courses",
        "area a required major courses",
    )
    for i in range(start, len(plain_lines)):
        normalized = normalize_for_match(plain_lines[i])
        if any(anchor in normalized for anchor in major_anchors):
            start = i
            break

    stop = len(plain_lines)
    stop_markers = (
        "other departmental program course requirements",
        "college of science core requirements",
        "non course non credit requirements",
        "university requirements",
        "sample 4 year plan",
    )
    for j in range(start, len(plain_lines)):
        normalized = normalize_for_match(plain_lines[j])
        if any(marker in normalized for marker in stop_markers):
            stop = j
            break

    scoped = plain_lines[start:stop]

    block_header_paren_re = re.compile(
        r"^(?P<title>[A-Za-z0-9&/:'(),\.\-*+ ]+?)\s*\((?P<credits>[^)]*credits?[^)]*)\)$",
        re.I,
    )
    block_header_credit_hours_re = re.compile(
        r"^(?P<title>[A-Za-z0-9&/:'(),\.\-*+ ≥]+?)\s*-\s*Credit Hours:\s*(?P<credits>.+)$",
        re.I,
    )

    blocks: list[dict[str, object]] = []
    active_title: str | None = None
    active_credits: str | None = None
    active_lines: list[str] = []

    def flush_block() -> None:
        nonlocal active_title, active_credits, active_lines
        if not active_title:
            return
        rules, courses, choose_count = build_rules_for_block(active_lines)
        blocks.append(
            {
                "title": active_title,
                "credits_text": active_credits,
                "choose_count_hint": choose_count,
                "rules": rules,
                "courses": [course_to_dict(c) for c in courses],
            }
        )
        active_title = None
        active_credits = None
        active_lines = []

    for line in scoped:
        m = block_header_paren_re.match(line) or block_header_credit_hours_re.match(line)
        if m:
            flush_block()
            active_title = normalize_text(m.group("title"))
            active_credits = normalize_text(m.group("credits"))
            active_lines = []
            continue

        if active_title:
            active_lines.append(line)

    flush_block()

    # keep only blocks that have at least one rule
    return [block for block in blocks if block["rules"]]


def default_output_paths(major: str, variant: str | None) -> tuple[Path, Path]:
    stem = slugify(major)
    if variant:
        stem = f"{stem}_{slugify(variant)}"
    json_path = OUT_DIR / f"{stem}_requirements_2024.json"
    md_path = OUT_DIR / f"{stem}_requirements_2024.md"
    return json_path, md_path


def render_rule(rule: dict[str, object]) -> str:
    rule_type = rule.get("type")
    if rule_type == "course":
        c = rule["course"]
        return f"Required: `{c['course_code']}` {c['title']} ({float(c['credits']):g})"
    if rule_type == "all_of":
        courses = rule["courses"]
        text = " + ".join(f"`{c['course_code']}`" for c in courses)
        return f"Take all: {text}"
    if rule_type == "choice_group":
        options = []
        for opt in rule["options"]:
            cs = opt["courses"]
            if len(cs) == 1:
                c = cs[0]
                options.append(f"`{c['course_code']}` {c['title']} ({float(c['credits']):g})")
            else:
                options.append(" + ".join(f"`{c['course_code']}`" for c in cs))
        return f"Choose {rule['choose']} from: " + "; ".join(options)
    return "Unknown rule"


def write_markdown(md_path: Path, output: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append(f"# {output['selected_heading']}")
    lines.append("")
    lines.append(f"- Source PDF: `{output['source_pdf']}`")
    lines.append(f"- Page window scanned: {output['section_pages']['start']} to {output['section_pages']['end_inclusive']}")
    lines.append("")

    blocks = output.get("major_requirement_blocks", [])
    if not blocks:
        lines.append("No requirement blocks parsed.")
    else:
        for block in blocks:
            lines.append(f"## {block['title']} ({block['credits_text']})")
            lines.append("")
            for rule in block["rules"]:
                lines.append(f"- {render_rule(rule)}")
            lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pdf_path: Path = args.pdf

    if not pdf_path.exists():
        print(f"[error] PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    lines = extract_lines(pdf_path)
    headings = find_degree_headings(lines)

    if args.list_headings:
        for h in headings:
            variant = f" ({h.variant})" if h.variant else ""
            print(f"p{h.page}: {h.title}, {h.degree}{variant}")
        return 0

    try:
        selected, candidate_list = choose_heading(
            headings=headings,
            major=args.major,
            degree=args.degree,
            variant=args.variant,
            concentration=args.concentration,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("\nDetected headings (sample):", file=sys.stderr)
        for h in headings[:40]:
            variant = f" ({h.variant})" if h.variant else ""
            print(f"  p{h.page}: {h.title}, {h.degree}{variant}", file=sys.stderr)
        return 2

    if len(candidate_list) > 1:
        preview = ", ".join(f"p{c.page}:{c.line_text}" for c in candidate_list[:4])
        print(f"[warn] multiple heading matches; using best match: {selected.line_text} (page {selected.page})")
        print(f"[warn] top candidates: {preview}")

    end_line = find_section_end_line(lines, selected, headings)
    section_lines = [rec for rec in lines if selected.line_index <= rec.idx < end_line]

    blocks = extract_blocks(section_lines)

    json_out, md_out = default_output_paths(args.major, args.variant)
    if args.json_out:
        json_out = args.json_out
    if args.md_out:
        md_out = args.md_out

    section_start_page = section_lines[0].page if section_lines else selected.page
    section_end_page = section_lines[-1].page if section_lines else selected.page

    output = {
        "source_pdf": str(pdf_path),
        "target": {
            "major": args.major,
            "degree": args.degree,
            "variant": args.variant,
            "concentration": args.concentration,
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

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(md_out, output)

    print(f"Selected heading: {selected.line_text} (page {selected.page})")
    print(f"Parsed blocks: {len(blocks)}")
    print(f"Wrote JSON: {json_out}")
    print(f"Wrote Markdown: {md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
