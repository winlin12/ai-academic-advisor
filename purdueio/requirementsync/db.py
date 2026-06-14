from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from requirementsync.catalog_scraper import ParsedProgram


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
NAMESPACE = uuid.UUID("cb844f92-404e-45db-940c-28926d2f3e97")


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row)


def apply_migrations(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS advisor")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS advisor.schema_migrations (
              version text PRIMARY KEY,
              applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.stem
            cur.execute(
                "SELECT 1 FROM advisor.schema_migrations WHERE version = %s",
                (version,),
            )
            if cur.fetchone():
                continue
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute(
                "INSERT INTO advisor.schema_migrations (version) VALUES (%s)",
                (version,),
            )
    conn.commit()


def stable_uuid(*parts: Any) -> uuid.UUID:
    key = "|".join("" if part is None else str(part) for part in parts)
    return uuid.uuid5(NAMESPACE, key)


def normalize_code(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def build_course_map(conn: psycopg.Connection) -> dict[str, uuid.UUID]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c."Id", s."Abbreviation", c."Number"
            FROM public."Courses" c
            JOIN public."Subjects" s ON s."Id" = c."SubjectId"
            WHERE s."Abbreviation" IS NOT NULL AND c."Number" IS NOT NULL
            """
        )
        rows = cur.fetchall()

    course_map: dict[str, uuid.UUID] = {}
    for row in rows:
        code = normalize_code(f"{row['Abbreviation']}{row['Number']}")
        if code and code not in course_map:
            course_map[code] = row["Id"]
    return course_map


def start_sync_run(
    conn: psycopg.Connection,
    *,
    catalog_year: int,
    catalog_label: str,
    catalog_base_url: str,
    metadata: dict[str, Any],
) -> uuid.UUID:
    run_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO advisor.requirement_sync_runs (
              id, catalog_year, catalog_label, catalog_base_url, status, metadata
            ) VALUES (%s, %s, %s, %s, 'running', %s)
            """,
            (run_id, catalog_year, catalog_label, catalog_base_url, Jsonb(metadata)),
        )
    conn.commit()
    return run_id


def finish_sync_run(
    conn: psycopg.Connection,
    *,
    run_id: uuid.UUID,
    status: str,
    programs_found: int,
    programs_loaded: int,
    errors: list[dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE advisor.requirement_sync_runs
            SET status = %s,
                completed_at = now(),
                programs_found = %s,
                programs_loaded = %s,
                errors = %s
            WHERE id = %s
            """,
            (status, programs_found, programs_loaded, Jsonb(errors), run_id),
        )
    conn.commit()


def replace_catalog_year(
    conn: psycopg.Connection,
    *,
    catalog_year: int,
    run_id: uuid.UUID,
    programs: list[ParsedProgram],
    course_map: dict[str, uuid.UUID],
) -> int:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM advisor.degree_programs WHERE catalog_year = %s",
                (catalog_year,),
            )
            for program in programs:
                insert_program(cur, run_id=run_id, program=program, course_map=course_map)
    return len(programs)


def insert_program(
    cur: psycopg.Cursor,
    *,
    run_id: uuid.UUID,
    program: ParsedProgram,
    course_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    program_id = stable_uuid("program", program.catalog_year, program.source_url)
    cur.execute(
        """
        INSERT INTO advisor.degree_programs (
          id, catalog_year, school, program_title, degree_code, variant, source_url,
          source_hash, parser_status, raw_json, sync_run_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            program_id,
            program.catalog_year,
            program.school,
            program.program_title,
            program.degree_code,
            program.variant,
            program.source_url,
            program.source_hash,
            program.parser_status,
            Jsonb(program.to_raw_json()),
            run_id,
        ),
    )

    for block_index, block in enumerate(program.blocks, start=1):
        insert_block(cur, program_id, block_index, block, course_map)
    return program_id


def insert_block(
    cur: psycopg.Cursor,
    program_id: uuid.UUID,
    block_index: int,
    block: dict[str, Any],
    course_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    block_id = stable_uuid(program_id, "block", block_index, block.get("title"))
    cur.execute(
        """
        INSERT INTO advisor.degree_requirement_blocks (
          id, program_id, sort_order, title, credits_text, raw_json
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            block_id,
            program_id,
            block_index,
            block.get("title"),
            block.get("credits_text"),
            Jsonb(block),
        ),
    )

    rules = block.get("rules") if isinstance(block.get("rules"), list) else []
    for rule_index, rule in enumerate(rules, start=1):
        if isinstance(rule, dict):
            insert_rule(cur, block_id, rule_index, rule, course_map)
    return block_id


def insert_rule(
    cur: psycopg.Cursor,
    block_id: uuid.UUID,
    rule_index: int,
    rule: dict[str, Any],
    course_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    rule_id = stable_uuid(block_id, "rule", rule_index, rule.get("raw_text"))
    cur.execute(
        """
        INSERT INTO advisor.degree_requirement_rules (
          id, block_id, sort_order, rule_type, choose_count, raw_text, raw_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            rule_id,
            block_id,
            rule_index,
            rule.get("rule_type") or "unknown",
            rule.get("choose_count"),
            rule.get("raw_text"),
            Jsonb(rule),
        ),
    )

    options = rule.get("options") if isinstance(rule.get("options"), list) else []
    for option_index, option in enumerate(options, start=1):
        if isinstance(option, dict):
            insert_option(cur, rule_id, option_index, option, course_map)
    return rule_id


def insert_option(
    cur: psycopg.Cursor,
    rule_id: uuid.UUID,
    option_index: int,
    option: dict[str, Any],
    course_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    option_id = stable_uuid(rule_id, "option", option_index, option.get("label"))
    cur.execute(
        """
        INSERT INTO advisor.degree_requirement_options (
          id, rule_id, option_index, sort_order, label, raw_json
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            option_id,
            rule_id,
            option_index,
            option_index,
            option.get("label"),
            Jsonb(option),
        ),
    )

    courses = option.get("courses") if isinstance(option.get("courses"), list) else []
    for course_index, course in enumerate(courses, start=1):
        if isinstance(course, dict):
            insert_option_course(cur, option_id, course_index, course, course_map)
    return option_id


def insert_option_course(
    cur: psycopg.Cursor,
    option_id: uuid.UUID,
    course_index: int,
    course: dict[str, Any],
    course_map: dict[str, uuid.UUID],
) -> uuid.UUID:
    course_code_text = str(course.get("course_code_text") or "").strip()
    normalized = normalize_code(str(course.get("normalized_code") or course_code_text))
    course_id = course_map.get(normalized)
    row_id = stable_uuid(option_id, "course", course_index, course_code_text)

    cur.execute(
        """
        INSERT INTO advisor.degree_requirement_courses (
          id, option_id, sort_order, course_code_text, course_id, course_title,
          credits_text, raw_text, raw_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row_id,
            option_id,
            course_index,
            course_code_text,
            course_id,
            course.get("course_title"),
            course.get("credits_text"),
            course.get("raw_text"),
            Jsonb(course),
        ),
    )
    return row_id


def json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
