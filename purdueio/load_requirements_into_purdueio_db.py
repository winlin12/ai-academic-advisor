#!/usr/bin/env python3
"""Load extracted major requirements into a Purdue.io-compatible SQLite database.

This script adds extension tables (degree programs + requirement graph) that are
joinable to Purdue.io's existing Courses table by CourseId when available.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SUMMARY = Path("purdueio/data/extracted/2024_all_requirements/summary.json")
DEFAULT_DB = Path("purdueio/purdue_api_academic.db")

NAMESPACE = uuid.UUID("f65d9cc1-6100-4f10-b8a0-0f8f7dc9a642")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load extracted major requirements into SQLite extension tables."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to summary.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Target SQLite DB")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing degree requirement extension rows before load.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def norm_code(value: str) -> str:
    return "".join(norm_text(value).upper().split())


def make_uuid(*parts: str) -> str:
    key = "|".join(parts)
    return str(uuid.uuid5(NAMESPACE, key))


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS DegreePrograms (
          Id TEXT PRIMARY KEY,
          CatalogYear INTEGER NOT NULL,
          School TEXT NOT NULL,
          ProgramTitle TEXT NOT NULL,
          DegreeCode TEXT NOT NULL,
          Variant TEXT,
          VariantKey TEXT NOT NULL,
          Heading TEXT,
          SourcePdf TEXT,
          SectionStartPage INTEGER,
          SectionEndPage INTEGER,
          RequirementsJsonPath TEXT,
          CreatedAt TEXT NOT NULL,
          UpdatedAt TEXT NOT NULL,
          UNIQUE(CatalogYear, School, ProgramTitle, DegreeCode, VariantKey)
        );

        CREATE TABLE IF NOT EXISTS DegreeRequirementBlocks (
          Id TEXT PRIMARY KEY,
          ProgramId TEXT NOT NULL,
          SortOrder INTEGER NOT NULL,
          Title TEXT,
          CreditsText TEXT,
          ChooseCountHint INTEGER,
          RawJson TEXT,
          CreatedAt TEXT NOT NULL,
          UpdatedAt TEXT NOT NULL,
          FOREIGN KEY (ProgramId) REFERENCES DegreePrograms(Id) ON DELETE CASCADE,
          UNIQUE(ProgramId, SortOrder)
        );

        CREATE TABLE IF NOT EXISTS DegreeRequirementRules (
          Id TEXT PRIMARY KEY,
          BlockId TEXT NOT NULL,
          SortOrder INTEGER NOT NULL,
          RuleType TEXT NOT NULL,
          ChooseCount INTEGER,
          RawJson TEXT,
          CreatedAt TEXT NOT NULL,
          UpdatedAt TEXT NOT NULL,
          FOREIGN KEY (BlockId) REFERENCES DegreeRequirementBlocks(Id) ON DELETE CASCADE,
          UNIQUE(BlockId, SortOrder)
        );

        CREATE TABLE IF NOT EXISTS DegreeRequirementRuleOptions (
          Id TEXT PRIMARY KEY,
          RuleId TEXT NOT NULL,
          OptionIndex INTEGER NOT NULL,
          SortOrder INTEGER NOT NULL,
          CreatedAt TEXT NOT NULL,
          UpdatedAt TEXT NOT NULL,
          FOREIGN KEY (RuleId) REFERENCES DegreeRequirementRules(Id) ON DELETE CASCADE,
          UNIQUE(RuleId, OptionIndex)
        );

        CREATE TABLE IF NOT EXISTS DegreeRequirementOptionCourses (
          Id TEXT PRIMARY KEY,
          OptionId TEXT NOT NULL,
          SortOrder INTEGER NOT NULL,
          CourseCode TEXT NOT NULL,
          CourseTitle TEXT,
          CourseCredits REAL,
          CourseId TEXT,
          RawJson TEXT,
          CreatedAt TEXT NOT NULL,
          UpdatedAt TEXT NOT NULL,
          FOREIGN KEY (OptionId) REFERENCES DegreeRequirementRuleOptions(Id) ON DELETE CASCADE,
          UNIQUE(OptionId, SortOrder)
        );

        CREATE INDEX IF NOT EXISTS IX_DegreePrograms_School ON DegreePrograms(School);
        CREATE INDEX IF NOT EXISTS IX_DegreePrograms_Title ON DegreePrograms(ProgramTitle);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementBlocks_ProgramId ON DegreeRequirementBlocks(ProgramId);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementRules_BlockId ON DegreeRequirementRules(BlockId);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementRuleOptions_RuleId ON DegreeRequirementRuleOptions(RuleId);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementOptionCourses_OptionId ON DegreeRequirementOptionCourses(OptionId);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementOptionCourses_CourseCode ON DegreeRequirementOptionCourses(CourseCode);
        CREATE INDEX IF NOT EXISTS IX_DegreeRequirementOptionCourses_CourseId ON DegreeRequirementOptionCourses(CourseId);
        """
    )


def clear_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM DegreeRequirementOptionCourses;
        DELETE FROM DegreeRequirementRuleOptions;
        DELETE FROM DegreeRequirementRules;
        DELETE FROM DegreeRequirementBlocks;
        DELETE FROM DegreePrograms;
        """
    )


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def build_course_map(conn: sqlite3.Connection) -> dict[str, str]:
    if not (table_exists(conn, "Courses") and table_exists(conn, "Subjects")):
        return {}

    rows = conn.execute(
        """
        SELECT c.Id, s.Abbreviation, c.Number
        FROM Courses c
        JOIN Subjects s ON s.Id = c.SubjectId
        """
    ).fetchall()

    course_map: dict[str, str] = {}
    for course_id, abbreviation, number in rows:
        if not abbreviation or not number:
            continue
        code = norm_code(f"{abbreviation}{number}")
        if code and code not in course_map:
            course_map[code] = str(course_id)
    return course_map


def resolve_course_id(course_code: str, course_map: dict[str, str]) -> str | None:
    code = norm_code(course_code)
    if not code:
        return None
    return course_map.get(code)


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def upsert_program(
    conn: sqlite3.Connection,
    *,
    program_id: str,
    catalog_year: int,
    school: str,
    title: str,
    degree: str,
    variant: str | None,
    heading: str,
    source_pdf: str,
    section_start: int | None,
    section_end: int | None,
    requirements_json_path: str,
    now: str,
) -> None:
    variant_key = norm_text(variant)
    conn.execute(
        """
        INSERT INTO DegreePrograms (
          Id, CatalogYear, School, ProgramTitle, DegreeCode, Variant, VariantKey,
          Heading, SourcePdf, SectionStartPage, SectionEndPage, RequirementsJsonPath,
          CreatedAt, UpdatedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Id) DO UPDATE SET
          CatalogYear = excluded.CatalogYear,
          School = excluded.School,
          ProgramTitle = excluded.ProgramTitle,
          DegreeCode = excluded.DegreeCode,
          Variant = excluded.Variant,
          VariantKey = excluded.VariantKey,
          Heading = excluded.Heading,
          SourcePdf = excluded.SourcePdf,
          SectionStartPage = excluded.SectionStartPage,
          SectionEndPage = excluded.SectionEndPage,
          RequirementsJsonPath = excluded.RequirementsJsonPath,
          UpdatedAt = excluded.UpdatedAt
        """,
        (
            program_id,
            catalog_year,
            school,
            title,
            degree,
            variant,
            variant_key,
            heading,
            source_pdf,
            section_start,
            section_end,
            requirements_json_path,
            now,
            now,
        ),
    )


def load_requirement_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Requirement file is not an object: {path}")
    return data


def insert_rule_option_courses(
    conn: sqlite3.Connection,
    *,
    option_id: str,
    option_sort: int,
    courses: list[dict[str, Any]],
    now: str,
    course_map: dict[str, str],
    counters: dict[str, int],
) -> None:
    for idx, course in enumerate(courses, start=1):
        course_code = norm_text(course.get("course_code"))
        if not course_code:
            continue

        option_course_id = make_uuid(option_id, "course", str(idx), course_code)
        course_id = resolve_course_id(course_code, course_map)
        if course_id:
            counters["course_links_resolved"] += 1
        else:
            counters["course_links_unresolved"] += 1

        conn.execute(
            """
            INSERT INTO DegreeRequirementOptionCourses (
              Id, OptionId, SortOrder, CourseCode, CourseTitle, CourseCredits,
              CourseId, RawJson, CreatedAt, UpdatedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Id) DO UPDATE SET
              OptionId = excluded.OptionId,
              SortOrder = excluded.SortOrder,
              CourseCode = excluded.CourseCode,
              CourseTitle = excluded.CourseTitle,
              CourseCredits = excluded.CourseCredits,
              CourseId = excluded.CourseId,
              RawJson = excluded.RawJson,
              UpdatedAt = excluded.UpdatedAt
            """,
            (
                option_course_id,
                option_id,
                idx,
                course_code,
                norm_text(course.get("title")),
                course.get("credits"),
                course_id,
                json.dumps(course),
                now,
                now,
            ),
        )
        counters["courses_inserted"] += 1


def process_rule(
    conn: sqlite3.Connection,
    *,
    rule_id: str,
    rule_sort: int,
    rule: dict[str, Any],
    block_id: str,
    now: str,
    course_map: dict[str, str],
    counters: dict[str, int],
) -> None:
    rule_type = norm_text(rule.get("type")) or "unknown"
    choose_count = rule.get("choose") if isinstance(rule.get("choose"), int) else None

    conn.execute(
        """
        INSERT INTO DegreeRequirementRules (
          Id, BlockId, SortOrder, RuleType, ChooseCount, RawJson, CreatedAt, UpdatedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Id) DO UPDATE SET
          BlockId = excluded.BlockId,
          SortOrder = excluded.SortOrder,
          RuleType = excluded.RuleType,
          ChooseCount = excluded.ChooseCount,
          RawJson = excluded.RawJson,
          UpdatedAt = excluded.UpdatedAt
        """,
        (
            rule_id,
            block_id,
            rule_sort,
            rule_type,
            choose_count,
            json.dumps(rule),
            now,
            now,
        ),
    )
    counters["rules_inserted"] += 1

    # Normalize into options + option courses for all rule types.
    options_payload: list[list[dict[str, Any]]] = []

    if rule_type == "course":
        course = rule.get("course")
        if isinstance(course, dict):
            options_payload = [[course]]
    elif rule_type == "all_of":
        courses = [c for c in ensure_list(rule.get("courses")) if isinstance(c, dict)]
        if courses:
            options_payload = [courses]
    elif rule_type == "choice_group":
        parsed: list[list[dict[str, Any]]] = []
        for option in ensure_list(rule.get("options")):
            if not isinstance(option, dict):
                continue
            courses = [c for c in ensure_list(option.get("courses")) if isinstance(c, dict)]
            if courses:
                parsed.append(courses)
        options_payload = parsed

    for option_idx, option_courses in enumerate(options_payload, start=1):
        option_id = make_uuid(rule_id, "option", str(option_idx))
        conn.execute(
            """
            INSERT INTO DegreeRequirementRuleOptions (
              Id, RuleId, OptionIndex, SortOrder, CreatedAt, UpdatedAt
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(Id) DO UPDATE SET
              RuleId = excluded.RuleId,
              OptionIndex = excluded.OptionIndex,
              SortOrder = excluded.SortOrder,
              UpdatedAt = excluded.UpdatedAt
            """,
            (option_id, rule_id, option_idx, option_idx, now, now),
        )
        counters["options_inserted"] += 1

        insert_rule_option_courses(
            conn,
            option_id=option_id,
            option_sort=option_idx,
            courses=option_courses,
            now=now,
            course_map=course_map,
            counters=counters,
        )


def process_requirement_payload(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    catalog_year: int,
    school: str,
    requirements_json_path: str,
    now: str,
    course_map: dict[str, str],
    counters: dict[str, int],
) -> None:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    title = norm_text(target.get("major"))
    degree = norm_text(target.get("degree"))
    variant = norm_text(target.get("variant")) or None
    heading = norm_text(payload.get("selected_heading"))
    source_pdf = norm_text(payload.get("source_pdf"))

    section_pages = payload.get("section_pages") if isinstance(payload.get("section_pages"), dict) else {}
    section_start = section_pages.get("start") if isinstance(section_pages.get("start"), int) else None
    section_end = (
        section_pages.get("end_inclusive")
        if isinstance(section_pages.get("end_inclusive"), int)
        else None
    )

    if not title or not degree:
        raise RuntimeError("Missing target.major or target.degree")

    program_id = make_uuid(
        "program",
        str(catalog_year),
        school,
        title,
        degree,
        variant or "",
    )

    upsert_program(
        conn,
        program_id=program_id,
        catalog_year=catalog_year,
        school=school,
        title=title,
        degree=degree,
        variant=variant,
        heading=heading,
        source_pdf=source_pdf,
        section_start=section_start,
        section_end=section_end,
        requirements_json_path=requirements_json_path,
        now=now,
    )
    counters["programs_inserted"] += 1

    blocks = [b for b in ensure_list(payload.get("major_requirement_blocks")) if isinstance(b, dict)]
    for block_idx, block in enumerate(blocks, start=1):
        block_title = norm_text(block.get("title"))
        block_id = make_uuid(program_id, "block", str(block_idx), block_title)

        choose_hint = block.get("choose_count_hint") if isinstance(block.get("choose_count_hint"), int) else None

        conn.execute(
            """
            INSERT INTO DegreeRequirementBlocks (
              Id, ProgramId, SortOrder, Title, CreditsText, ChooseCountHint,
              RawJson, CreatedAt, UpdatedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Id) DO UPDATE SET
              ProgramId = excluded.ProgramId,
              SortOrder = excluded.SortOrder,
              Title = excluded.Title,
              CreditsText = excluded.CreditsText,
              ChooseCountHint = excluded.ChooseCountHint,
              RawJson = excluded.RawJson,
              UpdatedAt = excluded.UpdatedAt
            """,
            (
                block_id,
                program_id,
                block_idx,
                block_title,
                norm_text(block.get("credits_text")),
                choose_hint,
                json.dumps(block),
                now,
                now,
            ),
        )
        counters["blocks_inserted"] += 1

        rules = [r for r in ensure_list(block.get("rules")) if isinstance(r, dict)]
        for rule_idx, rule in enumerate(rules, start=1):
            rule_id = make_uuid(block_id, "rule", str(rule_idx), norm_text(rule.get("type")))
            process_rule(
                conn,
                rule_id=rule_id,
                rule_sort=rule_idx,
                rule=rule,
                block_id=block_id,
                now=now,
                course_map=course_map,
                counters=counters,
            )


def main() -> None:
    args = parse_args()

    if not args.summary.exists():
        raise SystemExit(f"Summary file not found: {args.summary}")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise SystemExit("Summary file is not a JSON object")

    catalog_year = summary.get("year")
    if not isinstance(catalog_year, int):
        raise SystemExit("Summary JSON missing integer 'year'")

    schools = summary.get("schools")
    if not isinstance(schools, list):
        raise SystemExit("Summary JSON missing 'schools' list")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        create_tables(conn)

        if args.clear:
            clear_tables(conn)
            conn.commit()

        course_map = build_course_map(conn)
        print(f"Course map entries available: {len(course_map)}")

        now = utc_now()
        counters = {
            "programs_inserted": 0,
            "blocks_inserted": 0,
            "rules_inserted": 0,
            "options_inserted": 0,
            "courses_inserted": 0,
            "course_links_resolved": 0,
            "course_links_unresolved": 0,
            "files_processed": 0,
            "files_missing": 0,
            "files_failed": 0,
        }

        summary_dir = args.summary.parent

        for school in schools:
            if not isinstance(school, dict):
                continue
            school_name = norm_text(school.get("school"))
            majors = school.get("majors")
            if not isinstance(majors, list):
                continue

            for major in majors:
                if not isinstance(major, dict):
                    continue

                req_json_path = major.get("requirements_json")
                if not isinstance(req_json_path, str) or not req_json_path.strip():
                    continue

                req_path = Path(req_json_path)
                if not req_path.is_absolute():
                    req_path = Path.cwd() / req_path
                if not req_path.exists():
                    alt = summary_dir / Path(req_json_path).name
                    if alt.exists():
                        req_path = alt

                if not req_path.exists():
                    counters["files_missing"] += 1
                    print(f"[missing] {req_json_path}")
                    continue

                try:
                    payload = load_requirement_file(req_path)
                    process_requirement_payload(
                        conn,
                        payload=payload,
                        catalog_year=catalog_year,
                        school=school_name,
                        requirements_json_path=str(req_path),
                        now=now,
                        course_map=course_map,
                        counters=counters,
                    )
                    counters["files_processed"] += 1
                except Exception as exc:  # noqa: BLE001
                    counters["files_failed"] += 1
                    print(f"[error] {req_path}: {exc}")

        conn.commit()

        print("\nLoad complete.")
        print(json.dumps(counters, indent=2))
        print(f"Database: {args.db}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
