"""A mock of the advisor's real databases, and the whole-database context document built from it.

WHY THIS EXISTS. Until now the course catalog, the prerequisite edges and the degree
requirements were formatted into the Mode B prompt by hand — `catalog_block()` rendered a
bespoke bullet list that exists nowhere in production. That measured a prompt the app will
never send. In production the model is fed *retrieved database rows*, and the shape of those
rows is decided by the ingestion schema, not by a prompt author.

So this module holds a mock database that mirrors the real schemas row for row, and renders it
as the context document the model actually sees. What is being measured becomes "can the model
plan from our data as it is stored", which is the question that transfers.

TWO REAL DATABASES ARE MIRRORED, and keeping their boundary visible matters, because the two
halves have completely different provenance and reliability:

    catalog_ingestion (postgres `catalog_ingestion`) — crawled from Purdue's Acalog catalog
        catalog_years, colleges, departments, subjects, courses, course_aliases,
        programs, requirement_groups (self-referencing tree), requirement_options, program_notes

    purdueio, `advisor` schema — everything Acalog does NOT publish
        course_planner_terms       observed from PurdueIO Classes rows (002_course_offerings);
                                   the VIEW the planner queries, over course_offering_patterns
        course_prerequisites       crawled from Banner course-detail pages (003_course_prerequisites)
        course_workload            advisor-assigned, not sourced from anywhere upstream

That split is reproduced in `source` on every table below, and it is printed in the export, so
a model reading "confidence: high" next to a prerequisite knows it came from a page that stated
it — and the report's caveat that prereq edges are the highest-risk rows in the corpus stays
true of the mock exactly as it is of the real thing.

NO RETRIEVAL YET, ON PURPOSE. `render_context()` returns the ENTIRE database. That is only
tenable because a single-program mock is a few thousand tokens; a real catalog is ~10,000
courses and would not fit in any context. When RAG lands, this function is the thing that gets
replaced by a retrieval step, and the prompt around it does not have to change.

THE MOCK IS GENERATED, NOT HAND-MAINTAINED. `run.py build-mock-db` projects it out of the plan
fixture, which is and remains the scoring authority — so there is exactly one place to edit a
prerequisite, and `run.py check` fails if the two have drifted. Hand-editing these JSON files
is therefore a mistake that the next build silently reverts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Table -> which real database it mirrors. Printed in the export header; also the list that
# decides what `load_mock_db` expects to find on disk, so a table added here without a file is
# a loud failure rather than a silently absent section of the model's context.
TABLES: dict[str, str] = {
    "catalog_years": "catalog_ingestion",
    "programs": "catalog_ingestion",
    "requirement_groups": "catalog_ingestion",
    "requirement_options": "catalog_ingestion",
    "courses": "catalog_ingestion",
    "course_aliases": "catalog_ingestion",
    "program_notes": "catalog_ingestion",
    "course_prerequisites": "purdueio.advisor",
    "course_planner_terms": "purdueio.advisor",
    "course_workload": "purdueio.advisor",
}

# Columns the real tables have that the mock deliberately drops, and why. Stated in the export
# so the omission is visible to the reader (and to anyone diffing this against the real schema)
# rather than looking like the mock is the whole truth.
OMITTED_COLUMNS = (
    "uuid primary keys and foreign keys (natural keys — course_code, program id slugs — are "
    "used instead); provenance columns (source_page_id, sync_run_id, updated_at, term_code); "
    "columns whose value is derivable from course_code (subject, number); per-season "
    "observation counts (fall_terms/spring_terms/summer_terms — only the total, "
    "terms_observed, survives the fixture); courses.description, which in the real database "
    "holds Purdue's catalog prose; and courses.prerequisites_raw / corequisites_raw, which are "
    "empty in the real database too — Acalog publishes neither, which is the entire reason "
    "course_prerequisites exists as a separate, separately-crawled table"
)

# Rows are printed one per line, compact. A pretty-printed export of the same rows is 2.5x the
# tokens for no added information: at indent=1 this database rendered to ~13,300 tokens, which
# does not fit the prompt budget at all, against ~5,000 like this. Every value still carries its
# column name, which is the property that actually helps a small model.
#
# ensure_ascii=False everywhere: the program names carry em-dashes, and \u2014 is three tokens
# of nothing. The model reads UTF-8 fine.
_ROW_SEPARATORS = (",", ":")


@dataclass
class MockDatabase:
    path: Path
    # table name -> rows, in the order the file stores them (which is the order the export
    # prints and therefore part of the static prompt — never re-sort at render time).
    tables: dict[str, list[dict[str, Any]]]
    db_hash: str
    # Set by `load_mock_db` from the on-disk bytes of every table file, so it is the hash of
    # what was actually read.
    files: dict[str, str] = field(default_factory=dict)

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.tables.get(table, [])

    def one(self, table: str, **match: Any) -> dict[str, Any] | None:
        for row in self.rows(table):
            if all(row.get(k) == v for k, v in match.items()):
                return row
        return None

    def where(self, table: str, **match: Any) -> list[dict[str, Any]]:
        return [r for r in self.rows(table)
                if all(r.get(k) == v for k, v in match.items())]

    @property
    def course_codes(self) -> set[str]:
        return {r["course_code"] for r in self.rows("courses")}


def load_mock_db(root: Path) -> MockDatabase:
    """Read every table in TABLES from ``root/<table>.json``.

    The hash is over the raw bytes of every file, in table order — the same discipline as
    ``fixture_hash``. It travels into the Mode B static prompt (via the rendered export), so a
    change to the mock database invalidates Mode B comparisons exactly the way editing the
    system prompt does.
    """
    tables: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for table in TABLES:
        path = Path(root) / f"{table}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"mock database is missing {path.name}. Run `python run.py build-mock-db` to "
                f"regenerate it from the plan fixture."
            )
        raw = path.read_bytes()
        digest.update(raw)
        files[table] = hashlib.sha256(raw).hexdigest()[:16]
        rows = json.loads(raw.decode("utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path.name} must hold a JSON array of rows, got {type(rows)}")
        tables[table] = rows
    return MockDatabase(path=Path(root), tables=tables,
                        db_hash=digest.hexdigest()[:16], files=files)


def integrity_problems(db: MockDatabase) -> list[str]:
    """Referential integrity the real database would enforce with foreign keys.

    Natural keys buy readability at the cost of the constraint, so the constraint is checked
    here instead. Every one of these would otherwise show up as a model being scored against a
    requirement whose course it was never shown.
    """
    problems: list[str] = []
    codes = db.course_codes
    program_ids = {p["id"] for p in db.rows("programs")}
    group_ids = {g["id"] for g in db.rows("requirement_groups")}

    for group in db.rows("requirement_groups"):
        if group["program_id"] not in program_ids:
            problems.append(f"requirement_groups.{group['id']} -> unknown program "
                            f"{group['program_id']}")
        parent = group.get("parent_group_id")
        if parent is not None and parent not in group_ids:
            problems.append(f"requirement_groups.{group['id']} -> unknown parent {parent}")

    for option in db.rows("requirement_options"):
        if option["requirement_group_id"] not in group_ids:
            problems.append(f"requirement_options -> unknown group "
                            f"{option['requirement_group_id']}")
        if option["course_code"] not in codes:
            problems.append(f"requirement_options -> unknown course {option['course_code']}")

    for table, column in (("course_prerequisites", "course_code"),
                          ("course_planner_terms", "course_code"),
                          ("course_workload", "course_code"),
                          ("course_aliases", "course_code")):
        for row in db.rows(table):
            if row[column] not in codes:
                problems.append(f"{table} -> unknown course {row[column]}")

    for row in db.rows("course_prerequisites"):
        for group in row.get("prereq_groups") or []:
            for code in group:
                if code not in codes:
                    problems.append(f"course_prerequisites.{row['course_code']} requires "
                                    f"{code}, which is not in courses")
        for code in row.get("coreq_codes") or []:
            if code not in codes:
                problems.append(f"course_prerequisites.{row['course_code']} co-requires "
                                f"{code}, which is not in courses")
    for row in db.rows("course_aliases"):
        if row["alias_of"] not in codes:
            problems.append(f"course_aliases.{row['course_code']} -> unknown alias_of "
                            f"{row['alias_of']}")

    # Every course must be offered SOMETIME. A course with no observed terms is unschedulable,
    # and the deterministic planner would strand it without ever explaining why.
    for row in db.rows("course_planner_terms"):
        if not row.get("offered_terms"):
            problems.append(f"course_planner_terms.{row['course_code']} is offered in no "
                            f"season — nothing could ever schedule it")
    missing = codes - {r["course_code"] for r in db.rows("course_planner_terms")}
    if missing:
        problems.append(f"no offering pattern for: {', '.join(sorted(missing))}")
    return problems


def chain_depths(db: MockDatabase) -> dict[str, int]:
    """Longest prerequisite chain behind each course. 0 = nothing has to come first.

    Computed rather than stored, exactly as the app would compute it after retrieval. It is the
    one derived number in the export, and the system prompt is careful to say what it is NOT:
    a year, a class standing, or a semester index. See the note in `render_context`.
    """
    # A course with no prerequisites has NO ROW in course_prerequisites (see the export
    # header), so absence here means "no edges", not "unknown" — hence the {} default rather
    # than skipping the code entirely.
    known = db.course_codes
    prereqs: dict[str, list[str]] = {code: [] for code in known}
    for row in db.rows("course_prerequisites"):
        prereqs[row["course_code"]] = [
            code for group in (row.get("prereq_groups") or []) for code in group
        ]

    depth: dict[str, int] = {}
    resolving: set[str] = set()

    def resolve(code: str) -> int:
        if code in depth:
            return depth[code]
        # Cycle-safe: a course being resolved counts as 0 while in progress, so a bad edge
        # cannot hang prompt building. `run.py check` is what reports the bad edge.
        if code in resolving or code not in prereqs:
            return 0
        resolving.add(code)
        # Membership tested against the COURSE universe, not against `prereqs`' keys. Testing
        # the keys was wrong the moment prereq-less courses stopped having rows: every edge
        # pointing at one (CS 18000, MA 16100 — the roots of every chain) was silently dropped,
        # which flattened the whole graph to depth 0 and destroyed the ordering the export
        # relies on.
        value = max((resolve(p) + 1 for p in prereqs[code] if p in known), default=0)
        resolving.discard(code)
        depth[code] = value
        return value

    for code in known:
        resolve(code)
    return depth


def _export_rows(db: MockDatabase, table: str, depths: dict[str, int]) -> list[dict[str, Any]]:
    """Rows as exported: stored columns, plus the derived ones the app would compute."""
    rows = [dict(row) for row in db.rows(table)]
    if table == "course_prerequisites":
        for row in rows:
            row["chain_depth"] = depths.get(row["course_code"], 0)
    return rows


def render_context(db: MockDatabase) -> str:
    """The whole database, as the model sees it. Deterministic — this lands in a static hash.

    JSON per table rather than aligned text tables: every value carries its column name with it,
    so a small model never has to track a header row across forty lines to find out which field
    it is reading. It is also the shape a retrieval layer will hand over when RAG replaces this
    function, so the prompt around it will not have to change.

    Row order is FILE order and is never re-sorted here. `build-mock-db` emits courses in
    prerequisite-chain order, which means simply reading the export top to bottom is close to a
    legal scheduling order — the property the old `[level N]` prefix was trying to convey, now
    carried by the ordering itself instead of by a number that reads like a class year.
    """
    depths = chain_depths(db)
    parts = [
        "DATABASE EXPORT — advisor catalog snapshot, read-only. One JSON row per line.",
        "",
        "This is the complete contents of the advisor's catalog database for this student's "
        "program. Every course that exists is in `courses`; nothing outside this export exists.",
        "",
        "Sources, which differ in reliability:",
        "  - catalog_ingestion : crawled from Purdue's official course catalog.",
        "  - purdueio.advisor  : what that catalog does not publish. `course_planner_terms` "
        "is OBSERVED — a course is offered in fall because it HAS RUN in falls, and "
        "`terms_observed` is how many terms of history that is based on (0 = never observed, "
        "the terms are an editorial guess). `course_prerequisites` is read from course-detail "
        "pages and carries a `confidence`.",
        "",
        f"Not in this export: {OMITTED_COLUMNS}.",
        "",
        "How to read the less obvious columns:",
        "  - A course with NO ROW in `course_prerequisites` has no prerequisites and no "
        "corequisites. Every course here was crawled, so absence is a fact, not a gap.",
        "  - Every course is fixed-credit: `credit_hours_min` is the credit value.",
        "  - course_prerequisites.prereq_groups is AND-of-ORs: [[\"CS 25000\"],[\"CS 25100\","
        "\"CS 25300\"]] means CS 25000 AND (CS 25100 OR CS 25300). [] means no prerequisites.",
        "  - course_prerequisites.coreq_codes may be taken in the SAME semester or earlier.",
        "  - course_prerequisites.chain_depth is COMPUTED, not stored: how many courses deep "
        "the longest prerequisite chain behind this one runs. It is NOT a year, a class "
        "standing, or a semester number — chain_depth 3 does not mean 'junior year' or "
        "'semester 3', it means three courses must be completed in sequence first, which can "
        "happen at any point in the plan. `courses` is listed in this order, so a course never "
        "appears above one it depends on.",
        "  - Every requirement_options row requires a minimum grade of C; join it to `courses` "
        "on course_code for the title and credits.",
        "  - course_workload.workload_score is an advisor estimate, 1 (light) to 5 (heavy). It "
        "is not catalog data and no rule depends on it.",
        "  - course_aliases.alias_of names an approved substitute: the two codes satisfy the "
        "same requirement and the same prerequisites, so take one, never both.",
        "",
    ]
    for table, source in TABLES.items():
        rows = _export_rows(db, table, depths)
        parts.append(f"## {table}  ({source}, {len(rows)} rows)")
        parts += [json.dumps(row, sort_keys=True, separators=_ROW_SEPARATORS,
                             ensure_ascii=False) for row in rows]
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
