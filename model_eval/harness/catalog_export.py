"""The row shape the advisor's real databases are read into, and the context document built
from it.

WHY THIS EXISTS. Until Mode B had a real data source, the course catalog, the prerequisite
edges and the degree requirements were formatted into the prompt by hand — a bespoke bullet
list that existed nowhere in production. That measured a prompt the app would never send. In
production the model is fed *retrieved database rows*, and the shape of those rows is decided
by the ingestion schema, not by a prompt author — so this module holds that shape
(`CatalogDatabase`, table by table) and renders it as the context document the model actually
sees (`render_context`). What is being measured is "can the model plan from our data as it is
stored", which is the question that transfers.

USED TO BE BACKED BY A HAND-GENERATED JSON MOCK, one program's worth, projected out of the
plan fixture. It is not any more — `harness/real_db.py` is the only loader now, reading these
same rows straight from the real `catalog_ingestion`/`advisor` Postgres tables. The mock
existed because a whole-catalog export (~10,000 courses) does not fit any context window;
real_db.py's answer to that is per-program scoping plus budget-aware trimming, not a hand-kept
subset — see its module docstring. This module never knew which of the two it was rendering,
which is exactly why swapping the source cost one file.

TWO REAL DATABASES ARE MIRRORED, and keeping their boundary visible matters, because the two
halves have completely different provenance and reliability:

    catalog_ingestion (postgres `catalog_ingestion`) — crawled from Purdue's Acalog catalog
        catalog_years, colleges, departments, subjects, courses, course_aliases,
        programs, requirement_groups (self-referencing tree), requirement_options, program_notes

    purdueio, `advisor` schema — everything Acalog does NOT publish
        course_planner_terms       observed from PurdueIO Classes rows (002_course_offerings);
                                   the VIEW the planner queries, over course_offering_patterns
        course_prerequisites       crawled from Banner course-detail pages (003_course_prerequisites)

That split is reproduced in `source` on every table below, and it is printed in the export, so
a model reading "confidence: high" next to a prerequisite knows it came from a page that
stated it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Table -> which real database it mirrors. Printed in the export header; also the list
# `real_db.load_real_db` reads off to know what to fetch, so a table added here without a
# matching query there is a loud KeyError rather than a silently absent section of the export.
#
# `course_prerequisites` WAS a separate table here, one row per course-with-prereqs, until
# 2026-08-06: every row repeated `course_code` (already the join key back to `courses`) plus an
# object wrapper, for data that is 1:1 with a course and never queried on its own. Folded
# directly into each `courses` row instead — `prereq_groups`/`coreq_codes`/`confidence`/
# `chain_depth` appear on the course itself, present only when that course has prerequisites
# (see `_export_rows`).
#
# `course_planner_terms` is REMOVED entirely, not merged — the "not every course runs every
# term" constraint it fed both the prompt and the scorer with. Purdue's own published offering
# pattern is not always reliable prediction of a future term, and the harness was scoring
# models against a constraint that carries real uncertainty as if it were a hard fact. Every
# consumer (the prompt text, `plan_scorers`/`real_scoring`'s `term_offering_violation`,
# `convergence.py`'s `legal_slots`) reads `offered_terms` defensively (`if offered_terms and
# ...`), so simply not populating it here is what turns the constraint off everywhere at once,
# without touching any scoring code.
#
# `course_workload` is REMOVED entirely, 2026-08-07 — same class of problem as
# `course_planner_terms`, one step further along: it was never even an OBSERVATION, just a
# single advisor-assigned 1-5 number with no upstream source at all, presented next to real
# crawled/observed tables as if it belonged among them. Workload isn't a property of a course in
# the first place — it varies by professor, and by the same professor across terms — so no
# single per-course number was ever going to be honest. Nothing scored against it (it was
# explicitly "not catalog data, no rule depends on it" in the prompt text itself); it only ever
# cost tokens and offered the model a plausible-looking number to reason from that meant nothing.
TABLES: dict[str, str] = {
    "catalog_years": "catalog_ingestion",
    "programs": "catalog_ingestion",
    "requirement_groups": "catalog_ingestion",
    "requirement_options": "catalog_ingestion",
    "unresolved_requirement_groups": "catalog_ingestion",
    "courses": "catalog_ingestion",
    "course_aliases": "catalog_ingestion",
    "program_notes": "catalog_ingestion",
}

# Columns the real tables have that this export deliberately drops, and why. Stated in the
# export so the omission is visible to the reader (and to anyone diffing this against the real
# schema) rather than looking like the export is the whole truth.
OMITTED_COLUMNS = (
    "uuid primary keys and foreign keys (natural keys — course_code, program id — are used "
    "instead); provenance columns (source_page_id, sync_run_id, updated_at, term_code); "
    "columns whose value is derivable from course_code (subject, number); "
    "courses.description, which holds Purdue's catalog prose; and courses.prerequisites_raw / "
    "corequisites_raw, which Acalog does not publish — the entire reason a separately-crawled "
    "prerequisite source exists at all. Also, as of 2026-08-06, course offering-term data "
    "entirely: see the module docstring on why the harness stopped scoring against it."
)

# Rows are printed one per line, compact. A pretty-printed export of the same rows is ~2.5x the
# tokens for no added information. Every value still carries its column name, which is the
# property that actually helps a small model.
#
# ensure_ascii=False everywhere: program names carry em-dashes, and — is three tokens of
# nothing. The model reads UTF-8 fine.
_ROW_SEPARATORS = (",", ":")


@dataclass
class CatalogDatabase:
    """One program's worth of catalog rows, in the shape `render_context` prints.

    Source-agnostic on purpose: `real_db.load_real_db` is the only thing that builds one today,
    reading real Postgres, but nothing here knows that — `.tables` is a plain
    `dict[str, list[dict]]`, so a test can build one by hand without a database at all.
    """

    path: Path
    # table name -> rows, in the order they print (never re-sorted at render time — see
    # `real_db.py`'s ordering notes, particularly that `courses` is prerequisite-chain order).
    tables: dict[str, list[dict[str, Any]]]
    db_hash: str
    # table -> a short hash of just that table's rows, for per-table drift diagnostics.
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


def integrity_problems(db: CatalogDatabase) -> list[str]:
    """Referential integrity the real database enforces with foreign keys, re-checked here
    against the natural-key rows this export actually carries.

    Real Postgres data should never fail these — they exist as a safety net for `real_db.py`
    bugs, not because the source data is expected to be inconsistent. Every one of these would
    otherwise show up as a model being scored against a requirement whose course it was never
    shown.
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

    for table, column in (("course_aliases", "course_code"),):
        for row in db.rows(table):
            if row[column] not in codes:
                problems.append(f"{table} -> unknown course {row[column]}")

    # `course_prerequisites` is folded into `courses` now (see the module docstring) —
    # `prereq_groups`/`coreq_codes` are checked on the course row itself, not a separate table.
    for row in db.rows("courses"):
        for group in row.get("prereq_groups") or []:
            for code in group:
                if code not in codes:
                    problems.append(f"courses.{row['course_code']} requires "
                                    f"{code}, which is not in courses")
        for code in row.get("coreq_codes") or []:
            if code not in codes:
                problems.append(f"courses.{row['course_code']} co-requires "
                                f"{code}, which is not in courses")
    for row in db.rows("course_aliases"):
        if row["alias_of"] not in codes:
            problems.append(f"course_aliases.{row['course_code']} -> unknown alias_of "
                            f"{row['alias_of']}")
    return problems


def chain_depths(db: CatalogDatabase) -> dict[str, int]:
    """Longest prerequisite chain behind each course. 0 = nothing has to come first.

    Computed rather than stored, exactly as the app would compute it after retrieval. It is the
    one derived number in the export, and the system prompt is careful to say what it is NOT:
    a year, a class standing, or a semester index. See the note in `render_context`.
    """
    # A course with no prerequisites carries NO `prereq_groups` key on its `courses` row (see
    # the export header), so absence here means "no edges", not "unknown" — hence the {}
    # default rather than skipping the code entirely.
    known = db.course_codes
    prereqs: dict[str, list[str]] = {
        row["course_code"]: [code for group in (row.get("prereq_groups") or []) for code in group]
        for row in db.rows("courses")
    }

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


def _export_rows(db: CatalogDatabase, table: str, depths: dict[str, int]) -> list[dict[str, Any]]:
    """Rows as exported: stored columns, plus the derived ones the app would compute.

    `courses` gets `chain_depth` added to EVERY row (0 for a course with no prerequisites, same
    as any other) — `prereq_groups`/`coreq_codes`/`confidence` were already merged into the row
    by `real_db.py` and are simply absent for a course with neither, not zeroed out, so a
    prereq-less course's line stays exactly as short as `courses`-only fields would make it.
    """
    rows = [dict(row) for row in db.rows(table)]
    if table == "courses":
        for row in rows:
            row["chain_depth"] = depths.get(row["course_code"], 0)
    return rows


def render_context(db: CatalogDatabase) -> str:
    """The program's catalog, as the model sees it. Deterministic for a given database state —
    this lands in a static prompt hash.

    JSON per table rather than aligned text tables: every value carries its column name with
    it, so a small model never has to track a header row across forty lines to find out which
    field it is reading. It is also the shape a retrieval layer would hand over if this ever
    became one, so the prompt around it would not have to change.

    Row order is TABLE order and is never re-sorted here — `real_db.py` builds `courses` in
    prerequisite-chain order, which means simply reading the export top to bottom is close to a
    legal scheduling order, and every other table is built to match it.
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
        "  - purdueio.advisor  : what that catalog does not publish. Prerequisite/corequisite "
        "fields on `courses` are read from course-detail pages and carry a `confidence`.",
        "",
        f"Not in this export: {OMITTED_COLUMNS}.",
        "",
        "How to read the less obvious columns:",
        "  - A course with no `prereq_groups`/`coreq_codes` fields has no prerequisites and no "
        "corequisites. Every course here was crawled, so absence is a fact, not a gap.",
        "  - Every course is fixed-credit: `credit_hours_min` is the credit value.",
        "  - courses.prereq_groups is AND-of-ORs: [[\"CS 25000\"],[\"CS 25100\","
        "\"CS 25300\"]] means CS 25000 AND (CS 25100 OR CS 25300).",
        "  - courses.coreq_codes may be taken in the SAME semester or earlier.",
        "  - courses.chain_depth is COMPUTED, not stored: how many courses deep the longest "
        "prerequisite chain behind this one runs. It is NOT a year, a class standing, or a "
        "semester number — chain_depth 3 does not mean 'junior year' or 'semester 3', it means "
        "three courses must be completed in sequence first, which can happen at any point in "
        "the plan. `courses` is listed in this order, so a course never appears above one it "
        "depends on.",
        "  - There is no offering-term data in this export — Purdue's published pattern is not "
        "always a reliable predictor of a future term, so place courses by prerequisite order "
        "only; do not assume or invent a term restriction for any course.",
        "  - Every requirement_options row requires a minimum grade of C; join it to `courses` "
        "on course_code for the title and credits.",
        "  - Courses in `courses` that are not named in any `requirement_options` row are "
        "ELECTIVES: real courses in this department, not tied to a specific requirement. They "
        "count toward a selective/elective group's credit target same as any listed option.",
        "  - course_aliases.alias_of names an approved substitute: the two codes satisfy the "
        "same requirement and the same prerequisites, so take one, never both.",
        "  - unresolved_requirement_groups are real degree requirements — university-wide "
        "gen-ed, college/school core, world language — that the catalog states in prose "
        "instead of as a course list. There is nothing to schedule for them: never invent a "
        "course code to satisfy one, and \"cover every requirement group\" refers only to "
        "`requirement_groups`, not to these. `scope` says whose requirement it is — "
        "`university` (every student), `college` (every student in that school), or `program` "
        "(this degree only).",
        "",
    ]
    for table, source in TABLES.items():
        rows = _export_rows(db, table, depths)
        parts.append(f"## {table}  ({source}, {len(rows)} rows)")
        parts += [json.dumps(row, sort_keys=True, separators=_ROW_SEPARATORS,
                             ensure_ascii=False) for row in rows]
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_context_lean(db: CatalogDatabase) -> str:
    """The stripped-down export for a `program.source: fixture` pseudo-major (school-core /
    gen-ed+world-language test cases) — see `real_db.real_db_base_from_fixture`'s docstring for
    why those fixtures exist. Shows ONLY `requirement_groups`, each with its options joined
    inline as (code, credits) pairs — no separate `courses` table, no title, no prerequisites,
    no confidence, no chain_depth.

    Deliberately, not an oversight: these domains (gen-ed, world language, a college's own
    first-year orientation gate) have no meaningful prerequisite chains to model, and the whole
    point of splitting them out of the full major fixtures is to stop asking a model to do
    catalog-scale course selection AND real prerequisite sequencing in the same call — see the
    duplicate-course investigation that led here. `render_context`'s full per-course metadata
    would just be noise for a requirement domain that is, by construction, shallow.

    Deterministic for a given database state, same as `render_context` — lands in the static
    prompt hash the same way.
    """
    options_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in db.rows("requirement_options"):
        options_by_group.setdefault(row["requirement_group_id"], []).append(
            {"code": row["course_code"], "credits": row.get("credits")}
        )
    parts = [
        "REQUIREMENT GROUPS — this student's remaining degree requirements for this test case. "
        "One JSON row per group; no separate course catalog.",
        "",
        "This is the complete set of requirement groups here; nothing outside this export "
        "exists — never invent a course code to satisfy a requirement.",
        "",
        "How to read a row:",
        "  - kind \"all\": every listed option is required.",
        "  - kind \"choose\": pick options totaling at least `credits_min` credits from the "
        "listed options.",
        "  - `options` is a list of (course code, credits) pairs. Nothing else about a course "
        "(title, prerequisites) is shown here on purpose — these requirements have no "
        "meaningful prerequisite chain to sequence.",
        "",
    ]
    for row in db.rows("requirement_groups"):
        parts.append(json.dumps({
            "name": row.get("name"),
            "kind": "all" if row.get("requirement_type") == "all_of" else "choose",
            "credits_min": row.get("credits_min"),
            "options": options_by_group.get(row.get("id"), []),
        }, sort_keys=True, separators=_ROW_SEPARATORS, ensure_ascii=False))
    return "\n".join(parts).rstrip() + "\n"
