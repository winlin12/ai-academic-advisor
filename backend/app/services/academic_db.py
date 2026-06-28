"""Read access to the academic catalog.

Backed by the ``catalog_ingestion`` database (the Purdue catalog ingestion pipeline).
Its schema is:

    catalog_years ── programs ── requirement_groups (self-referencing tree)
                                      └── requirement_options ── courses
                  └── colleges, subjects, courses, source_pages

The richer ingestion schema is mapped onto the API's existing response models so the
HTTP contract (and any client) stays unchanged:

    degree_program          -> programs (+ catalog_years, colleges, source_pages)
    requirement *block*     -> a top-level requirement_group (parent_group_id IS NULL)
    requirement *rule*      -> each group in that block's subtree
    rule *option* / course  -> the group's requirement_options
"""

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


def _catalog_schema_ready(connection: psycopg.Connection) -> bool:
    """True once the ingestion schema (the programs table) exists."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.programs') AS table_name")
        row = cursor.fetchone()
    return bool(row and row["table_name"])


def _credits_text(credits_min: Any, credits_max: Any) -> str | None:
    """Render a credit range as text, e.g. '128', '78-79', or None."""
    if credits_min is None and credits_max is None:
        return None

    def _fmt(value: Any) -> str:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)

    if credits_min is not None and credits_max is not None and float(credits_min) != float(credits_max):
        return f"{_fmt(credits_min)}-{_fmt(credits_max)}"
    return _fmt(credits_min if credits_min is not None else credits_max)


def fetch_academic_facets() -> AcademicFacetResponse:
    with _get_connection() as connection:
        if not _catalog_schema_ready(connection):
            return AcademicFacetResponse(catalog_years=[], schools=[], subjects=[])

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT start_year FROM catalog_years ORDER BY start_year DESC"
            )
            years = [row["start_year"] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT DISTINCT name
                FROM colleges
                WHERE name IS NOT NULL AND btrim(name) <> ''
                ORDER BY name
                """
            )
            schools = [row["name"] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT DISTINCT code
                FROM subjects
                WHERE code IS NOT NULL AND btrim(code) <> ''
                ORDER BY code
                """
            )
            subjects = [row["code"] for row in cursor.fetchall()]

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
              p.name ILIKE %s
              OR COALESCE(col.name, '') ILIKE %s
              OR COALESCE(p.degree_type, '') ILIKE %s
            )
            """
        )
        params.extend([like_query, like_query, like_query])

    if catalog_year is not None:
        where_clauses.append("cy.start_year = %s")
        params.append(catalog_year)

    if school and school.strip():
        where_clauses.append("col.name = %s")
        params.append(school.strip())

    sql = """
        SELECT
          p.id::text AS id,
          cy.start_year AS catalog_year,
          col.name AS school,
          p.name AS program_title,
          p.degree_type AS degree_code,
          p.program_type AS variant,
          COALESCE(sp.url, '') AS source_url,
          COUNT(DISTINCT rg.id) FILTER (WHERE rg.parent_group_id IS NULL)::int AS block_count,
          COUNT(DISTINCT ro.id)::int AS course_count,
          COUNT(DISTINCT ro.course_id)::int AS linked_course_count
        FROM programs p
        JOIN catalog_years cy ON cy.id = p.catalog_year_id
        LEFT JOIN colleges col ON col.id = p.college_id
        LEFT JOIN source_pages sp ON sp.id = p.source_page_id
        LEFT JOIN requirement_groups rg ON rg.program_id = p.id
        LEFT JOIN requirement_options ro ON ro.requirement_group_id = rg.id
    """

    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"

    sql += """
        GROUP BY p.id, cy.start_year, col.name, p.name, p.degree_type, p.program_type, sp.url
        ORDER BY cy.start_year DESC, col.name ASC NULLS LAST, p.name ASC
        LIMIT %s
    """
    params.append(limit)

    with _get_connection() as connection:
        if not _catalog_schema_ready(connection):
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
            parser_status="parsed",
            block_count=row["block_count"],
            course_count=row["course_count"],
            linked_course_count=row["linked_course_count"],
        )
        for row in rows
    ]


def fetch_program_detail(program_id: str) -> AcademicProgramDetail | None:
    with _get_connection() as connection:
        if not _catalog_schema_ready(connection):
            return None

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                  p.id::text AS id,
                  cy.start_year AS catalog_year,
                  col.name AS school,
                  p.name AS program_title,
                  p.degree_type AS degree_code,
                  p.program_type AS variant,
                  COALESCE(sp.url, '') AS source_url
                FROM programs p
                JOIN catalog_years cy ON cy.id = p.catalog_year_id
                LEFT JOIN colleges col ON col.id = p.college_id
                LEFT JOIN source_pages sp ON sp.id = p.source_page_id
                WHERE p.id = %s
                """,
                (program_id,),
            )
            program_row = cursor.fetchone()
            if program_row is None:
                return None

            cursor.execute(
                """
                SELECT
                  rg.id::text AS group_id,
                  rg.parent_group_id::text AS parent_group_id,
                  rg.name AS group_name,
                  rg.requirement_type,
                  rg.credits_min,
                  rg.credits_max,
                  rg.raw_text AS group_raw_text,
                  rg.display_order AS group_order,
                  ro.id::text AS option_id,
                  ro.display_order AS option_order,
                  ro.course_code_raw,
                  ro.course_id::text AS resolved_course_id,
                  ro.course_title_raw,
                  ro.option_text,
                  ro.credits,
                  ro.credits_max AS option_credits_max,
                  ro.is_selective_option,
                  ro.minimum_grade
                FROM requirement_groups rg
                LEFT JOIN requirement_options ro ON ro.requirement_group_id = rg.id
                WHERE rg.program_id = %s
                ORDER BY rg.display_order ASC, ro.display_order ASC NULLS LAST
                """,
                (program_id,),
            )
            rows = cursor.fetchall()

    blocks = _build_blocks(rows)

    return AcademicProgramDetail(
        id=program_row["id"],
        catalog_year=program_row["catalog_year"],
        school=program_row["school"],
        program_title=program_row["program_title"],
        degree_code=program_row["degree_code"],
        variant=program_row["variant"],
        source_url=program_row["source_url"],
        parser_status="parsed",
        blocks=blocks,
    )


def _build_blocks(rows: list[dict[str, Any]]) -> list[RequirementBlockDetail]:
    """Fold the (group, option) rows into the block -> rule -> option -> course shape.

    Each top-level requirement_group becomes a block; every group in its subtree becomes
    a rule whose single option lists that group's courses. Narrative groups (GPA, policy,
    disclaimer) survive as rules carrying ``raw_text`` with no courses.
    """
    groups: dict[str, dict[str, Any]] = {}
    child_ids: dict[str, list[str]] = {}
    order: list[str] = []

    for row in rows:
        gid = row["group_id"]
        if gid not in groups:
            groups[gid] = {
                "id": gid,
                "parent": row["parent_group_id"],
                "name": row["group_name"],
                "requirement_type": row["requirement_type"],
                "credits_min": row["credits_min"],
                "credits_max": row["credits_max"],
                "raw_text": row["group_raw_text"],
                "order": row["group_order"] or 0,
                "courses": [],
            }
            order.append(gid)
            child_ids.setdefault(row["parent_group_id"], []).append(gid)

        if row["option_id"] is not None:
            groups[gid]["courses"].append(
                RequirementCourseOption(
                    id=row["option_id"],
                    sort_order=row["option_order"] or 0,
                    course_code_text=row["course_code_raw"] or row["option_text"] or "",
                    course_id=row["resolved_course_id"],
                    course_title=row["course_title_raw"],
                    credits_text=_credits_text(row["credits"], row["option_credits_max"]),
                    raw_text=row["option_text"],
                )
            )

    def subtree(root_id: str) -> list[str]:
        """Depth-first ids in the subtree rooted at root_id (root first), in display order."""
        result = [root_id]
        for cid in sorted(child_ids.get(root_id, []), key=lambda i: groups[i]["order"]):
            result.extend(subtree(cid))
        return result

    def make_rule(gid: str) -> RequirementRuleDetail:
        g = groups[gid]
        selective = bool(g["courses"]) and g["requirement_type"] in {
            "selective", "elective", "free_elective"
        }
        options: list[RequirementRuleOption] = []
        if g["courses"]:
            options.append(
                RequirementRuleOption(
                    id=f"{gid}-opt",
                    option_index=0,
                    sort_order=0,
                    label=g["name"],
                    courses=g["courses"],
                )
            )
        return RequirementRuleDetail(
            id=gid,
            sort_order=g["order"],
            rule_type=("choose" if selective else (g["requirement_type"] or "requirement")),
            choose_count=None,
            raw_text=g["raw_text"],
            options=options,
        )

    blocks: list[RequirementBlockDetail] = []
    top_level = [gid for gid in order if groups[gid]["parent"] is None]
    for gid in sorted(top_level, key=lambda i: groups[i]["order"]):
        g = groups[gid]
        rules = [make_rule(child) for child in subtree(gid)]
        blocks.append(
            RequirementBlockDetail(
                id=gid,
                sort_order=g["order"],
                title=g["name"],
                credits_text=_credits_text(g["credits_min"], g["credits_max"]),
                rules=rules,
            )
        )
    return blocks


def search_courses(
    *,
    query: str | None = None,
    subject: str | None = None,
    limit: int = 80,
) -> list[AcademicCourseResult]:
    where_clauses: list[str] = []
    params: list[Any] = []

    if subject and subject.strip():
        where_clauses.append("c.subject_code = %s")
        params.append(subject.strip().upper())

    if query and query.strip():
        like_query = f"%{query.strip()}%"
        where_clauses.append(
            """
            (
              c.title ILIKE %s
              OR COALESCE(c.description, '') ILIKE %s
              OR c.course_number ILIKE %s
              OR c.course_code ILIKE %s
            )
            """
        )
        params.extend([like_query, like_query, like_query, like_query])

    sql = """
        SELECT
          c.id::text AS id,
          c.subject_code AS subject,
          c.course_number AS number,
          c.title AS title,
          c.credit_hours_min AS credit_hours,
          c.description AS description
        FROM courses c
    """

    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"

    sql += " ORDER BY c.subject_code ASC, c.course_number ASC, c.title ASC LIMIT %s"
    params.append(limit)

    with _get_connection() as connection:
        if not _catalog_schema_ready(connection):
            return []
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

    return [
        AcademicCourseResult(
            id=row["id"],
            subject=row["subject"] or "N/A",
            number=row["number"] or "",
            code=f"{row['subject'] or 'N/A'} {row['number'] or ''}".strip(),
            title=row["title"] or "Untitled course",
            credit_hours=row["credit_hours"],
            description=row["description"],
        )
        for row in rows
    ]
