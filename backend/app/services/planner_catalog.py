"""Bridge the deterministic planner to the real academic catalog (§4 of TODO.md).

``generate_plan(profile, catalog)`` keeps its signature; this module builds that ``catalog``
from the ``catalog_ingestion`` database instead of the bundled 8-course ``courses.json``
fixture, and (when the profile carries a ``program_id``) derives ``remaining_courses`` from
the program's requirement rows instead of the client hand-listing codes.

TWO PATHS LIVE HERE, and the difference is whether a major was chosen:

* ``program_id`` SET → ``resolve_program`` delegates to :mod:`planner_db`, which returns the
  whole :class:`ProgramCatalog`: requirement groups, alias map, and courses carrying the real
  prerequisite edges and observed term offerings from the ``advisor``
  schema. This is the path the app takes, and the one Mode B needs.

* ``program_id`` ABSENT → the loose per-code path below. It looks the student's own course
  codes up in ``courses`` and can only produce the degraded ``Course`` that
  ``_course_from_row`` builds: no prereq edges and every term allowed, because Acalog
  publishes neither (TODO §2.5) and nothing here fabricates structure the source lacks. A plan
  built this way is a plan that believes CS 25100 can be taken in the first semester, which is
  exactly why choosing a major matters and why the AI planner refuses without one.

``courses.json`` remains the documented offline fixture: it is used whenever the academic
database is unreachable or knows none of the requested course codes, so the demo profile and
tests keep working without a database.
"""

from __future__ import annotations

import logging
import uuid

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import Course, StudentProfile
from app.services.catalog import load_catalog
from app.services.planner import TERM_ORDER
# ProgramNotFoundError is RE-EXPORTED, not redefined. ``deps.resolve_for_planning`` catches it
# by the name it imports from here, so there must be exactly one class — two identically-named
# exceptions would make the 404 mapping depend on which module happened to raise.
from app.services.planner_db import (
    ProgramCatalog,
    ProgramNotFoundError,
    load_program_catalog,
)
from app.services.planner_db import (
    select_remaining_courses as select_remaining_from_groups,
)

logger = logging.getLogger(__name__)

# requirement_group.requirement_type values that mean "choose from this list" rather than
# "take all of these" (mirrors the selective detection in academic_db._build_blocks).
SELECTIVE_TYPES = {"selective", "elective", "free_elective"}

# When a selective group gives no credit target and no option credits, assume a standard
# 3-credit course so "cover the group's credits" degrades to "choose one".
DEFAULT_OPTION_CREDITS = 3.0


def normalize_course_code(code: str) -> str:
    """Best-effort canonicalisation to the DB's ``'CS 18000'`` form.

    Accepts ``'cs18000'``, ``'CS  18000'``, ``'CS 18000'``; leaves codes with no digits
    untouched. Purely lexical — it never checks the code exists.
    """
    compact = "".join(code.split()).upper()
    for index, char in enumerate(compact):
        if char.isdigit():
            if index == 0:  # no subject prefix; nothing to split
                return compact
            return f"{compact[:index]} {compact[index:]}"
    return compact


def _course_from_row(row: dict) -> Course:
    credits = row["credit_hours_min"]
    return Course(
        code=row["course_code"],
        title=(row["title"] or "Untitled course").strip(),
        credits=max(int(round(float(credits))), 0) if credits is not None else 0,
        prereqs=[],
        offered_terms=list(TERM_ORDER),
        requirement_tags=[],
    )


def fetch_courses_by_codes(codes: set[str]) -> dict[str, Course]:
    """Fetch catalog courses as planner ``Course``s, keyed by *normalized* code.

    Prefers the newest catalog year when a code exists in several.
    """
    if not codes:
        return {}
    normalized = sorted({normalize_course_code(code) for code in codes})
    with psycopg.connect(settings.academic_database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (c.course_code)
                       c.course_code, c.title, c.credit_hours_min
                FROM courses c
                JOIN catalog_years cy ON cy.id = c.catalog_year_id
                WHERE c.course_code = ANY(%s)
                ORDER BY c.course_code, cy.start_year DESC
                """,
                (normalized,),
            )
            return {row["course_code"]: _course_from_row(row) for row in cur.fetchall()}


def _fetch_requirement_rows(program_id: str) -> list[dict] | None:
    """Requirement (group, option) rows for a program, or ``None`` if it doesn't exist."""
    with psycopg.connect(settings.academic_database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM programs WHERE id = %s", (program_id,))
            if cur.fetchone() is None:
                return None
            cur.execute(
                """
                SELECT
                  rg.id::text AS group_id,
                  rg.requirement_type,
                  rg.credits_min AS group_credits_min,
                  rg.display_order AS group_order,
                  COALESCE(c.course_code, ro.course_code_raw) AS course_code,
                  ro.credits AS option_credits,
                  ro.display_order AS option_order
                FROM requirement_groups rg
                JOIN requirement_options ro ON ro.requirement_group_id = rg.id
                LEFT JOIN courses c ON c.id = ro.course_id
                WHERE rg.program_id = %s
                ORDER BY rg.display_order ASC, ro.display_order ASC
                """,
                (program_id,),
            )
            return cur.fetchall()


def select_remaining_courses(rows: list[dict], completed: set[str]) -> list[str]:
    """Turn requirement (group, option) rows into an ordered remaining-course list.

    Required groups contribute every unmet course. Selective groups ("choose N from list")
    contribute just enough unmet options — in catalog display order — to cover the group's
    credit target, counting the student's completed courses toward the group first. Required
    blocks come before all selectives (TODO §4.3), and duplicates keep their first slot.
    """
    completed_normalized = {normalize_course_code(code) for code in completed}

    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(
            row["group_id"],
            {
                "selective": (row["requirement_type"] or "") in SELECTIVE_TYPES,
                "credits_min": row["group_credits_min"],
                "options": [],
            },
        )
        code = (row["course_code"] or "").strip()
        if code:
            group["options"].append(
                (normalize_course_code(code), row["option_credits"])
            )

    required: list[str] = []
    selective: list[str] = []
    seen: set[str] = set(completed_normalized)

    for group in groups.values():
        if group["selective"]:
            continue
        for code, _credits in group["options"]:
            if code not in seen:
                required.append(code)
                seen.add(code)

    for group in groups.values():
        if not group["selective"] or not group["options"]:
            continue
        target = group["credits_min"]
        if target is None:
            # No credit target on the group: treat it as "choose one".
            target = min(
                credits if credits is not None else DEFAULT_OPTION_CREDITS
                for _code, credits in group["options"]
            )
        remaining_needed = float(target) - sum(
            credits if credits is not None else DEFAULT_OPTION_CREDITS
            for code, credits in group["options"]
            if code in completed_normalized
        )
        for code, credits in group["options"]:
            if remaining_needed <= 0:
                break
            if code in seen:
                continue
            selective.append(code)
            seen.add(code)
            remaining_needed -= credits if credits is not None else DEFAULT_OPTION_CREDITS

    return required + selective


def derive_remaining_courses(program_id: str, completed: set[str]) -> list[str]:
    """Remaining courses for a program from the academic DB (raises ProgramNotFoundError)."""
    try:
        uuid.UUID(program_id)
    except ValueError:
        raise ProgramNotFoundError(program_id) from None
    rows = _fetch_requirement_rows(program_id)
    if rows is None:
        raise ProgramNotFoundError(program_id)
    return select_remaining_courses(rows, completed)


def resolve_program(profile: StudentProfile) -> tuple[StudentProfile, ProgramCatalog]:
    """Resolve a program-driven profile into (planner-ready profile, the program's catalog).

    THE MAJOR IS WHAT DECIDES THE COURSES. ``remaining_courses`` is derived from the selected
    program's requirement rows minus what the student has completed — any codes the client
    listed are ignored, because a plan for "Computer Science: Security" is defined by that
    program's requirement groups and not by what a form was pre-filled with.

    Two fields are derived from the program rather than asked of the student, because neither
    is a fact about a person: ``major_subject`` (the department contributing the most required
    courses) and ``degree_program`` (the catalog's own program title). A profile naming the
    wrong major subject silently disables the per-semester major-course cap, which is a limit
    the student would notice only by being handed four CS courses in one term.
    """
    catalog = load_program_catalog(
        profile.program_id or "", extra_codes=set(profile.completed_courses), profile=profile
    )
    completed = {normalize_course_code(code) for code in profile.completed_courses}
    remaining = select_remaining_from_groups(catalog.groups, completed, catalog.credits)
    profile = profile.model_copy(update={
        "completed_courses": sorted(completed),
        "remaining_courses": remaining,
        "major_subject": catalog.major_subject(),
        # The catalog's own title wins over whatever the client sent. StudentProfile defaults
        # `degree_program` to "Computer Science", so honouring the client value would print
        # that on a Mechanical Engineering plan for anyone whose form did not overwrite it.
        "degree_program": catalog.name,
    })
    return profile, catalog


def resolve_profile_and_catalog(profile: StudentProfile) -> tuple[StudentProfile, list[Course]]:
    """Resolve a request profile into (planner-ready profile, planner catalog).

    * ``program_id`` set → the program's own catalog answers, complete with the prerequisite
      edges and observed term offerings the ``advisor`` schema holds
      (see ``planner_db``). A missing program raises :class:`ProgramNotFoundError`; a database
      failure propagates (the caller maps it to 503) because a program-driven plan is
      meaningless without the DB.
    * Otherwise the profile's own codes are looked up in the academic DB; if it is unreachable
      or knows none of them, the bundled ``courses.json`` fixture answers instead, unchanged.
    """
    if profile.program_id:
        profile, catalog = resolve_program(profile)
        return profile, catalog.courses

    requested = set(profile.remaining_courses) | set(profile.completed_courses)
    try:
        catalog = fetch_courses_by_codes(requested)
    except psycopg.Error as exc:
        logger.warning("academic DB unavailable for planning; using bundled fixture: %s", exc)
        return profile, load_catalog()

    if not any(normalize_course_code(code) in catalog for code in profile.remaining_courses):
        return profile, load_catalog()

    # Keep the client's preference order, but rewrite matched codes to their canonical DB form
    # so the planner's catalog lookups (and the revise agent's course context) line up. Codes
    # the catalog doesn't know are kept as-is — the planner warns about them by name.
    def canonical(code: str) -> str:
        normalized = normalize_course_code(code)
        return catalog[normalized].code if normalized in catalog else code

    profile = profile.model_copy(
        update={
            "remaining_courses": [canonical(code) for code in profile.remaining_courses],
            "completed_courses": [
                normalize_course_code(code) for code in profile.completed_courses
            ],
        }
    )
    return profile, list(catalog.values())
