from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import (
    AcademicCourseResult,
    AcademicFacetResponse,
    AcademicProgramDetail,
    AcademicProgramSummary,
    RequirementBlockDetail,
    RequirementCourseOption,
    RequirementRuleDetail,
    RequirementRuleOption,
)


def _get_connection() -> psycopg.Connection:
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


def _advisor_schema_ready(connection: psycopg.Connection) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('advisor.degree_programs') AS table_name")
        row = cursor.fetchone()
    return bool(row and row["table_name"])


def fetch_academic_facets() -> AcademicFacetResponse:
    with _get_connection() as connection:
        years: list[int] = []
        schools: list[str] = []
        if _advisor_schema_ready(connection):
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT catalog_year
                    FROM advisor.degree_programs
                    ORDER BY catalog_year DESC
                    """
                )
                years = [row["catalog_year"] for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT DISTINCT school
                    FROM advisor.degree_programs
                    WHERE school IS NOT NULL AND btrim(school) <> ''
                    ORDER BY school
                    """
                )
                schools = [row["school"] for row in cursor.fetchall()]

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT "Abbreviation"
                FROM public."Subjects"
                WHERE "Abbreviation" IS NOT NULL AND btrim("Abbreviation") <> ''
                ORDER BY "Abbreviation"
                """
            )
            subjects = [row["Abbreviation"] for row in cursor.fetchall()]

    return AcademicFacetResponse(catalog_years=years, schools=schools, subjects=subjects)


def fetch_program_summaries(
    *,
    query: str | None = None,
    catalog_year: int | None = None,
    school: str | None = None,
    limit: int = 120,
) -> list[AcademicProgramSummary]:
    where_clauses: list[str] = []
    params: list[Any] = []

    if query and query.strip():
        like_query = f"%{query.strip()}%"
        where_clauses.append(
            """
            (
              p.program_title ILIKE %s
              OR COALESCE(p.school, '') ILIKE %s
              OR COALESCE(p.degree_code, '') ILIKE %s
            )
            """
        )
        params.extend([like_query, like_query, like_query])

    if catalog_year is not None:
        where_clauses.append("p.catalog_year = %s")
        params.append(catalog_year)

    if school and school.strip():
        where_clauses.append("p.school = %s")
        params.append(school.strip())

    sql = """
        SELECT
          p.id::text AS id,
          p.catalog_year,
          p.school,
          p.program_title,
          p.degree_code,
          p.variant,
          p.source_url,
          p.parser_status,
          COUNT(DISTINCT b.id)::int AS block_count,
          COUNT(DISTINCT rc.id)::int AS course_count,
          COUNT(DISTINCT rc.course_id)::int AS linked_course_count
        FROM advisor.degree_programs p
        LEFT JOIN advisor.degree_requirement_blocks b ON b.program_id = p.id
        LEFT JOIN advisor.degree_requirement_rules r ON r.block_id = b.id
        LEFT JOIN advisor.degree_requirement_options o ON o.rule_id = r.id
        LEFT JOIN advisor.degree_requirement_courses rc ON rc.option_id = o.id
    """

    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"

    sql += """
        GROUP BY p.id
        ORDER BY p.catalog_year DESC, p.school ASC NULLS LAST, p.program_title ASC
        LIMIT %s
    """
    params.append(limit)

    with _get_connection() as connection:
        if not _advisor_schema_ready(connection):
            return []
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

    return [
        AcademicProgramSummary(
            id=row["id"],
            catalog_year=row["catalog_year"],
            school=row["school"],
            program_title=row["program_title"],
            degree_code=row["degree_code"],
            variant=row["variant"],
            source_url=row["source_url"],
            parser_status=row["parser_status"],
            block_count=row["block_count"],
            course_count=row["course_count"],
            linked_course_count=row["linked_course_count"],
        )
        for row in rows
    ]


def fetch_program_detail(program_id: str) -> AcademicProgramDetail | None:
    with _get_connection() as connection:
        if not _advisor_schema_ready(connection):
            return None

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                  id::text,
                  catalog_year,
                  school,
                  program_title,
                  degree_code,
                  variant,
                  source_url,
                  parser_status
                FROM advisor.degree_programs
                WHERE id = %s
                """,
                (program_id,),
            )
            program_row = cursor.fetchone()
            if program_row is None:
                return None

            cursor.execute(
                """
                SELECT
                  b.id::text AS block_id,
                  b.sort_order AS block_sort_order,
                  b.title AS block_title,
                  b.credits_text AS block_credits_text,
                  r.id::text AS rule_id,
                  r.sort_order AS rule_sort_order,
                  r.rule_type,
                  r.choose_count,
                  r.raw_text AS rule_raw_text,
                  o.id::text AS option_id,
                  o.option_index,
                  o.sort_order AS option_sort_order,
                  o.label AS option_label,
                  rc.id::text AS course_option_id,
                  rc.sort_order AS course_sort_order,
                  rc.course_code_text,
                  rc.course_id::text AS resolved_course_id,
                  rc.course_title,
                  rc.credits_text AS course_credits_text,
                  rc.raw_text AS course_raw_text
                FROM advisor.degree_requirement_blocks b
                LEFT JOIN advisor.degree_requirement_rules r ON r.block_id = b.id
                LEFT JOIN advisor.degree_requirement_options o ON o.rule_id = r.id
                LEFT JOIN advisor.degree_requirement_courses rc ON rc.option_id = o.id
                WHERE b.program_id = %s
                ORDER BY
                  b.sort_order ASC,
                  r.sort_order ASC NULLS LAST,
                  o.sort_order ASC NULLS LAST,
                  rc.sort_order ASC NULLS LAST
                """,
                (program_id,),
            )
            rows = cursor.fetchall()

    blocks_by_id: dict[str, RequirementBlockDetail] = {}
    rules_by_id: dict[str, RequirementRuleDetail] = {}
    options_by_id: dict[str, RequirementRuleOption] = {}

    for row in rows:
        block_id = row["block_id"]
        if block_id not in blocks_by_id:
            blocks_by_id[block_id] = RequirementBlockDetail(
                id=block_id,
                sort_order=row["block_sort_order"],
                title=row["block_title"],
                credits_text=row["block_credits_text"],
                rules=[],
            )
        block = blocks_by_id[block_id]

        rule_id = row["rule_id"]
        if rule_id is None:
            continue
        if rule_id not in rules_by_id:
            rules_by_id[rule_id] = RequirementRuleDetail(
                id=rule_id,
                sort_order=row["rule_sort_order"],
                rule_type=row["rule_type"],
                choose_count=row["choose_count"],
                raw_text=row["rule_raw_text"],
                options=[],
            )
            block.rules.append(rules_by_id[rule_id])
        rule = rules_by_id[rule_id]

        option_id = row["option_id"]
        if option_id is None:
            continue
        if option_id not in options_by_id:
            options_by_id[option_id] = RequirementRuleOption(
                id=option_id,
                option_index=row["option_index"],
                sort_order=row["option_sort_order"],
                label=row["option_label"],
                courses=[],
            )
            rule.options.append(options_by_id[option_id])
        option = options_by_id[option_id]

        course_option_id = row["course_option_id"]
        if course_option_id is None:
            continue
        option.courses.append(
            RequirementCourseOption(
                id=course_option_id,
                sort_order=row["course_sort_order"],
                course_code_text=row["course_code_text"],
                course_id=row["resolved_course_id"],
                course_title=row["course_title"],
                credits_text=row["course_credits_text"],
                raw_text=row["course_raw_text"],
            )
        )

    return AcademicProgramDetail(
        id=program_row["id"],
        catalog_year=program_row["catalog_year"],
        school=program_row["school"],
        program_title=program_row["program_title"],
        degree_code=program_row["degree_code"],
        variant=program_row["variant"],
        source_url=program_row["source_url"],
        parser_status=program_row["parser_status"],
        blocks=list(blocks_by_id.values()),
    )


def search_courses(
    *,
    query: str | None = None,
    subject: str | None = None,
    limit: int = 80,
) -> list[AcademicCourseResult]:
    where_clauses: list[str] = []
    params: list[Any] = []

    if subject and subject.strip():
        where_clauses.append('s."Abbreviation" = %s')
        params.append(subject.strip().upper())

    if query and query.strip():
        like_query = f"%{query.strip()}%"
        where_clauses.append(
            """
            (
              c."Title" ILIKE %s
              OR COALESCE(c."Description", '') ILIKE %s
              OR c."Number" ILIKE %s
              OR COALESCE(s."Abbreviation", '') ILIKE %s
              OR (COALESCE(s."Abbreviation", '') || c."Number") ILIKE %s
              OR (COALESCE(s."Abbreviation", '') || ' ' || c."Number") ILIKE %s
            )
            """
        )
        params.extend([like_query, like_query, like_query, like_query, like_query, like_query])

    sql = """
        SELECT
          c."Id"::text AS id,
          COALESCE(s."Abbreviation", '') AS subject,
          c."Number" AS number,
          c."Title" AS title,
          c."CreditHours" AS credit_hours,
          c."Description" AS description
        FROM public."Courses" c
        LEFT JOIN public."Subjects" s ON s."Id" = c."SubjectId"
    """

    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"

    sql += ' ORDER BY subject ASC, c."Number" ASC, c."Title" ASC LIMIT %s'
    params.append(limit)

    with _get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

    return [
        AcademicCourseResult(
            id=row["id"],
            subject=row["subject"] or "N/A",
            number=row["number"] or "",
            code=f"{row['subject'] or 'N/A'}{row['number'] or ''}".replace(" ", ""),
            title=row["title"] or "Untitled course",
            credit_hours=row["credit_hours"],
            description=row["description"],
        )
        for row in rows
    ]
