"""Load a program's catalog into Postgres so the web app has something real to plan from.

WHAT PROBLEM THIS SOLVES. The `catalog_ingestion` database on this box is empty — every
table is 0 rows (the 2026-2027 crawl did not come with the machine, and `make restore` has
no backup file to restore from). Acalog's crawler honours a 120-second crawl-delay, so
re-crawling a degree is a multi-hour job, not a setup step. Meanwhile the app cannot show a
single plan without courses, requirement groups, prerequisite edges and term offerings.

The rows this loads are the ones `model_eval/mock_db/` already holds, and that is not a
shortcut — that directory exists precisely because it "mirrors the real schemas row for row"
(harness/mock_db.py), it is generated from `plan_fixtures/cs_machine_intelligence.yaml` which
is the eval's scoring authority, and its offering patterns were observed from PurdueIO Classes
rows rather than invented. So this is the same data the model was measured against, landing in
the real tables through the real column shapes.

WHAT IS AND IS NOT TRUSTWORTHY, carried over verbatim from the fixture's provenance header:

    course codes / titles / credits .. from the Purdue catalog. Spot-checkable.
    offered_terms .................... OBSERVED from PurdueIO Classes; `terms_observed` is
                                       the evidence count (0 = an editorial guess).
    prereq edges ..................... HAND-WRITTEN. Acalog publishes none and Banner's
                                       robots.txt disallows crawling, so these are the
                                       highest-risk rows in the corpus. Stored with
                                       confidence so a reader can see that.

None of it is official academic advice, and the fixture is marked `verified: false`. When a
real crawl lands, it writes the same tables through `catalog_ingestion`'s own ingest path and
this loader stops being the source — nothing downstream has to change, because everything
downstream reads Postgres, not these files.

Usage:
    python -m app.services.seed_catalog                 # load (idempotent upsert)
    python -m app.services.seed_catalog --source DIR    # from another mock_db export
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import psycopg

from app.core.config import settings

logger = logging.getLogger(__name__)

# model_eval/mock_db, relative to this file (app/services -> app -> backend -> repo root).
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "model_eval" / "mock_db"

SCHEMA_SQL = Path(__file__).resolve().parent / "advisor_schema.sql"

# Stable ids so re-running the loader updates rows instead of duplicating them. The natural
# keys (course code, program slug) are what the export actually carries; UUIDv5 turns them
# into the uuid primary keys the real schema uses, deterministically.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def _id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, "|".join(parts))


def split_code(course_code: str) -> tuple[str, str]:
    """'CS 25100' -> ('CS', '25100'). The subject/number columns are derivable from the code,
    which is why the export omits them (mock_db.OMITTED_COLUMNS) and why they are re-derived
    here rather than stored twice in the fixture."""
    compact = "".join(course_code.split()).upper()
    for index, char in enumerate(compact):
        if char.isdigit():
            return compact[:index], compact[index:]
    return compact, ""


def _load_tables(source: Path) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "catalog_years", "programs", "requirement_groups", "requirement_options",
        "courses", "course_aliases", "program_notes",
        "course_prerequisites", "course_planner_terms",
    ):
        path = source / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Regenerate the export with "
                f"`python model_eval/run.py build-mock-db`."
            )
        tables[name] = json.loads(path.read_text(encoding="utf-8"))
    return tables


def apply_advisor_schema(conn: psycopg.Connection) -> None:
    """Create the `advisor` schema tables. Idempotent; also called at API startup."""
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def _seed_catalog_year(cur: psycopg.Cursor, row: dict[str, Any]) -> uuid.UUID:
    year_id = _id("catalog_year", row["label"])
    cur.execute(
        """
        INSERT INTO catalog_years
            (id, label, catoid, start_year, end_year, is_archived, catalog_url,
             created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
        ON CONFLICT (id) DO UPDATE SET
            label = EXCLUDED.label, start_year = EXCLUDED.start_year,
            end_year = EXCLUDED.end_year, is_archived = EXCLUDED.is_archived,
            updated_at = now()
        """,
        (
            year_id, row["label"],
            # `catoid` is Acalog's own catalog id and has a UNIQUE constraint. Nothing crawled
            # it here, so it is derived from the start year — distinct per catalog year, and
            # obviously not an Acalog id to anyone reading it.
            int(row["start_year"]), int(row["start_year"]), int(row["end_year"]),
            bool(row.get("is_archived", False)),
            f"https://catalog.purdue.edu/index.php?catoid={row['start_year']}",
        ),
    )
    return year_id


def _seed_courses(
    cur: psycopg.Cursor, year_id: uuid.UUID, year_label: str, rows: list[dict[str, Any]]
) -> dict[str, uuid.UUID]:
    subjects = {split_code(row["course_code"])[0] for row in rows}
    for subject in sorted(subjects):
        cur.execute(
            """
            INSERT INTO subjects (id, catalog_year_id, code, name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (_id("subject", year_label, subject), year_id, subject, subject),
        )

    course_ids: dict[str, uuid.UUID] = {}
    for row in rows:
        code = row["course_code"]
        subject, number = split_code(code)
        course_id = _id("course", year_label, code)
        course_ids[code] = course_id
        cur.execute(
            """
            INSERT INTO courses
                (id, catalog_year_id, subject_id, subject_code, course_number, course_code,
                 title, credit_hours_min, credit_hours_max, attributes_raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                credit_hours_min = EXCLUDED.credit_hours_min,
                credit_hours_max = EXCLUDED.credit_hours_max,
                attributes_raw = EXCLUDED.attributes_raw
            """,
            (
                course_id, year_id, _id("subject", year_label, subject), subject, number, code,
                row.get("title"),
                float(row["credit_hours_min"]) if row.get("credit_hours_min") is not None else None,
                float(row["credit_hours_min"]) if row.get("credit_hours_min") is not None else None,
                # The requirement tags the planner reads ('required', 'theory-heavy', ...).
                # `attributes_raw` is the catalog's own free-text attribute column; storing JSON
                # in it keeps the tags queryable without inventing a column the crawler will
                # never populate. `planner_db` parses it back and tolerates real prose.
                json.dumps(row.get("attributes") or []),
            ),
        )
    return course_ids


def _seed_aliases(
    cur: psycopg.Cursor, course_ids: dict[str, uuid.UUID], rows: list[dict[str, Any]]
) -> None:
    """`alias_of` names the PRIMARY course; the row hangs off that primary's id.

    The export reads "MA 16500 is an approved substitute for MA 16100", and the real table
    reads "MA 16100 has alias code MA 16500" — same fact, opposite direction, so the join
    column is the primary's id and never the substitute's.
    """
    for row in rows:
        primary = course_ids.get(row["alias_of"])
        if primary is None:
            logger.warning("alias %s -> unknown course %s, skipped",
                           row["course_code"], row["alias_of"])
            continue
        cur.execute(
            """
            INSERT INTO course_aliases (id, course_id, alias_code, reason)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (course_id, alias_code) DO UPDATE SET reason = EXCLUDED.reason
            """,
            (_id("alias", row["alias_of"], row["course_code"]), primary,
             row["course_code"], row.get("reason")),
        )


def _seed_program(
    cur: psycopg.Cursor,
    year_id: uuid.UUID,
    course_ids: dict[str, uuid.UUID],
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, uuid.UUID]:
    program_ids: dict[str, uuid.UUID] = {}
    for row in tables["programs"]:
        program_id = _id("program", row["id"])
        program_ids[row["id"]] = program_id
        cur.execute(
            """
            INSERT INTO programs
                (id, catalog_year_id, name, degree_type, program_type, campus,
                 total_credits_min, total_credits_raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name, degree_type = EXCLUDED.degree_type,
                program_type = EXCLUDED.program_type, campus = EXCLUDED.campus,
                total_credits_min = EXCLUDED.total_credits_min
            """,
            (
                program_id, year_id, row["name"], row.get("degree_type"),
                row.get("program_type"), row.get("campus"),
                float(row["total_credits_min"]) if row.get("total_credits_min") else None,
                f"{row['total_credits_min']} credits" if row.get("total_credits_min") else None,
            ),
        )

    for index, row in enumerate(tables["program_notes"]):
        program_id = program_ids.get(row["program_id"])
        if program_id is None:
            continue
        cur.execute(
            """
            INSERT INTO program_notes (id, program_id, note_text, note_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                note_text = EXCLUDED.note_text, note_type = EXCLUDED.note_type
            """,
            (_id("note", row["program_id"], str(index)), program_id,
             row["note_text"], row.get("note_type")),
        )

    group_ids: dict[str, uuid.UUID] = {}
    for order, row in enumerate(tables["requirement_groups"]):
        program_id = program_ids.get(row["program_id"])
        if program_id is None:
            logger.warning("requirement group %s -> unknown program %s, skipped",
                           row["id"], row["program_id"])
            continue
        group_ids[row["id"]] = _id("group", row["id"])

    for order, row in enumerate(tables["requirement_groups"]):
        group_id = group_ids.get(row["id"])
        if group_id is None:
            continue
        cur.execute(
            """
            INSERT INTO requirement_groups
                (id, program_id, parent_group_id, name, requirement_type, credits_min,
                 raw_text, display_order)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name, requirement_type = EXCLUDED.requirement_type,
                credits_min = EXCLUDED.credits_min, raw_text = EXCLUDED.raw_text,
                display_order = EXCLUDED.display_order
            """,
            (
                group_id, program_ids[row["program_id"]],
                group_ids.get(row.get("parent_group_id")) if row.get("parent_group_id") else None,
                row.get("name"), row.get("requirement_type"),
                float(row["credits_min"]) if row.get("credits_min") is not None else None,
                row.get("raw_text"), order,
            ),
        )

    selective_types = {"choose_credits", "choose", "selective", "elective", "free_elective"}
    per_group_order: dict[str, int] = {}
    for row in tables["requirement_options"]:
        group_key = row["requirement_group_id"]
        group_id = group_ids.get(group_key)
        if group_id is None:
            logger.warning("requirement option -> unknown group %s, skipped", group_key)
            continue
        group = next(g for g in tables["requirement_groups"] if g["id"] == group_key)
        selective = (group.get("requirement_type") or "") in selective_types
        order = per_group_order.get(group_key, 0)
        per_group_order[group_key] = order + 1
        cur.execute(
            """
            INSERT INTO requirement_options
                (id, requirement_group_id, course_id, course_code_raw, credits,
                 minimum_grade, is_required, is_selective_option, display_order)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                course_id = EXCLUDED.course_id, credits = EXCLUDED.credits,
                is_required = EXCLUDED.is_required,
                is_selective_option = EXCLUDED.is_selective_option,
                display_order = EXCLUDED.display_order
            """,
            (
                _id("option", group_key, row["course_code"]), group_id,
                course_ids.get(row["course_code"]), row["course_code"],
                float(row["credits"]) if row.get("credits") is not None else None,
                # Every requirement_options row requires a minimum grade of C — stated once in
                # the export header rather than on all 49 rows, so it is re-attached here.
                "C",
                # `is_required` mirrors the group's kind. `is_selective_option` is left FALSE
                # unconditionally, and that is not the same question: in the crawled catalog it
                # marks an option whose text ends in "or", i.e. one chained to the NEXT row as
                # an alternative (MA 26100 or MA 27101). This export has no or-chains — its
                # substitutes live in `course_aliases` instead — so claiming otherwise would
                # make `planner_db._fetch_groups` fuse every option of a menu into a single
                # alternative and reduce the whole group to one course.
                not selective, False, order,
            ),
        )
    return program_ids


def _seed_advisor_tables(
    cur: psycopg.Cursor, year_label: str, tables: dict[str, list[dict[str, Any]]]
) -> None:
    for row in tables["course_prerequisites"]:
        code = row["course_code"]
        subject, number = split_code(code)
        groups = row.get("prereq_groups") or []
        coreqs = row.get("coreq_codes") or []
        cur.execute(
            """
            INSERT INTO advisor.course_prerequisites
                (course_code, subject, number, term_code, has_prereqs, raw_text,
                 prereq_groups, coreq_codes, confidence, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (course_code) DO UPDATE SET
                has_prereqs = EXCLUDED.has_prereqs, raw_text = EXCLUDED.raw_text,
                prereq_groups = EXCLUDED.prereq_groups, coreq_codes = EXCLUDED.coreq_codes,
                confidence = EXCLUDED.confidence, updated_at = now()
            """,
            (code, subject, number, year_label, bool(groups or coreqs), row.get("raw_text"),
             json.dumps(groups), json.dumps(coreqs), row.get("confidence") or "low"),
        )

    for row in tables["course_planner_terms"]:
        code = row["course_code"]
        subject, number = split_code(code)
        terms = set(row.get("offered_terms") or ())
        observed = int(row.get("terms_observed") or 0)
        cur.execute(
            """
            INSERT INTO advisor.course_offering_patterns
                (course_code, subject, number, offered_fall, offered_spring, offered_summer,
                 fall_terms, spring_terms, summer_terms, terms_observed, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (course_code) DO UPDATE SET
                offered_fall = EXCLUDED.offered_fall,
                offered_spring = EXCLUDED.offered_spring,
                offered_summer = EXCLUDED.offered_summer,
                fall_terms = EXCLUDED.fall_terms,
                spring_terms = EXCLUDED.spring_terms,
                summer_terms = EXCLUDED.summer_terms,
                terms_observed = EXCLUDED.terms_observed,
                updated_at = now()
            """,
            (
                code, subject, number,
                "fall" in terms, "spring" in terms, "summer" in terms,
                # The export keeps only the TOTAL observation count (per-season counts are in
                # mock_db.OMITTED_COLUMNS), so a season the course was seen in gets the total
                # and a season it was not gets zero. That preserves the one property callers
                # actually read — "is this pattern evidence or a guess" — without inventing
                # per-season numbers the export does not carry.
                observed if "fall" in terms else 0,
                observed if "spring" in terms else 0,
                observed if "summer" in terms else 0,
                observed,
            ),
        )

def seed(source: Path = DEFAULT_SOURCE, *, database_url: str | None = None) -> dict[str, int]:
    """Load the export at ``source`` into Postgres. Idempotent — safe to re-run."""
    tables = _load_tables(source)
    year_row = tables["catalog_years"][0]
    year_label = year_row["label"]

    with psycopg.connect(database_url or settings.academic_database_url) as conn:
        apply_advisor_schema(conn)
        with conn.cursor() as cur:
            year_id = _seed_catalog_year(cur, year_row)
            course_ids = _seed_courses(cur, year_id, year_label, tables["courses"])
            _seed_aliases(cur, course_ids, tables["course_aliases"])
            _seed_program(cur, year_id, course_ids, tables)
            _seed_advisor_tables(cur, year_label, tables)
        conn.commit()

    counts = {name: len(rows) for name, rows in tables.items()}
    logger.info("Seeded catalog from %s: %s", source, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"mock_db-shaped export directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--database-url", default=None,
                        help="override ACADEMIC_DATABASE_URL for this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = seed(args.source, database_url=args.database_url)
    width = max(len(name) for name in counts)
    print("Loaded into", args.database_url or settings.academic_database_url)
    for name, count in counts.items():
        print(f"  {name:<{width}}  {count:>4} rows")


if __name__ == "__main__":
    main()
