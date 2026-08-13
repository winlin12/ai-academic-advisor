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
# `requirement_options` is still FETCHED (real_db reads this dict to know what to query, and
# `real_scoring._groups_from_tables` scores off the flat rows) but as of 2026-08-12 it is no
# longer PRINTED as its own section: `_export_rows` folds each group's options into the
# `requirement_groups` row that owns them. This is a rendering change only — `.tables` keeps the
# flat rows exactly as before, so every scorer, `integrity_problems`, and `real_db`'s
# `_partial_database` are untouched.
#
# WHAT IT BUYS: the join column was a uuid repeated on every option row, and the group ids it
# pointed at were uuids too (some of them composite, `<uuid>:<uuid>`, for merged second-major and
# minor groups). Measured on the mi-fresh-start export with qwen3.6-27b's own tokenizer, uuids
# were 5,468 of the export's 12,077 tokens — 45% of the whole thing, spent on identifiers the
# model never reads back and never emits: plans are lists of course codes. Nesting removes the
# foreign key and the primary key together and takes the export to 5,472 tokens. Group ids that
# were readable slugs (`ucc-wc`, `merged-lab-science`) go too — a group is identified by its
# `name`, which is what the prompt and the scoring reports already call it by.
#
# `parent_group_id` goes with them, but is rendered as the parent's NAME when it is set, rather
# than dropped: the column is null on all 3,610 group rows across the recorded corpus, so this
# path is untested by construction and must not silently lose a tree if one ever appears.
#
# `unresolved_requirement_groups` is REMOVED, 2026-08-12, the same way and for a related
# reason: it was the one section of the export the model was shown and then told it could do
# NOTHING with ("never invent a course code to satisfy one", "cover every requirement group
# does not refer to these"), and the transcripts read like models trying to act on it anyway —
# prose requirements sitting next to schedulable ones is a distinction a small model is being
# asked to hold for no payoff. Nothing scored against it either (`real_scoring` only listed the
# rows back in the report). Dropping the table here turns it off in the export; the paired
# explanations in `render_context` and `prompts.PLAN_SYSTEM` came out with it, and `real_db` no
# longer fetches the rows. University requirements need a representation the model can act on —
# the resolved `ucc-*` groups `real_db` synthesizes are the shape that works — and that is the
# open piece of work this removal makes room for.
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
    "courses": "catalog_ingestion",
    "course_aliases": "catalog_ingestion",
    "program_notes": "catalog_ingestion",
}

# Columns the real tables have that this export deliberately drops, and why. Stated in the
# export so the omission is visible to the reader (and to anyone diffing this against the real
# schema) rather than looking like the export is the whole truth.
OMITTED_COLUMNS = (
    "uuid primary keys and foreign keys — the tables that referenced each other are printed "
    "already joined, so nothing here needs them (a course is named by course_code, a "
    "requirement group by its name); provenance columns (source_page_id, sync_run_id, "
    "updated_at, term_code); "
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

# Fetched into `CatalogDatabase.tables` and read by the scorers, but never printed as its own
# `## <table>` section — `_export_rows` folds it into the table that owns it. See the TABLES
# comment above for why.
NESTED_TABLES = frozenset({"requirement_options"})


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
    # CREDIT HOURS FOR THE WHOLE COURSE UNIVERSE, including courses trimmed out of `tables`.
    #
    # `courses` above is budget-trimmed to what fits one prompt, which is right for what the
    # model is SHOWN and wrong for what the student HAS. A completed gen-ed like ENGL 10600
    # routinely does not survive the trim, and scoring it off `tables` therefore counted zero
    # credits for a course the student really took — mi-ai-early lost 12 of its completed
    # credits that way (ENGL 10600, COM 11400, PHIL 11000, HIST 10300), which fed straight into
    # a false "short of the 120-credit graduation minimum". Populated from the UNTRIMMED base;
    # empty for a hand-built database, in which case scoring falls back to `tables` as before.
    all_credits: dict[str, float] = field(default_factory=dict)

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

    `requirement_groups` gets its `requirement_options` rows nested in as `options` and loses
    every uuid: its own `id`, the `program_id` naming the one program this whole export is for,
    and the `requirement_group_id` that was repeated on every option row. A group is identified
    by `name` from here on. `db.tables` is not touched — these are copies.
    """
    rows = [dict(row) for row in db.rows(table)]
    if table == "courses":
        for row in rows:
            row["chain_depth"] = depths.get(row["course_code"], 0)
    elif table == "programs":
        for row in rows:
            row.pop("id", None)
    elif table == "requirement_groups":
        options: dict[str, list[dict[str, Any]]] = {}
        for option in db.rows("requirement_options"):
            options.setdefault(option["requirement_group_id"], []).append(
                {k: v for k, v in option.items() if k != "requirement_group_id"}
            )
        names = {row["id"]: row.get("name") for row in rows}
        for row in rows:
            parent = row.pop("parent_group_id", None)
            row.pop("program_id", None)
            row["options"] = options.get(row["id"], [])
            # Name, not id, and only when set — see the TABLES comment: null everywhere in the
            # recorded corpus, so losing it silently is the failure mode to avoid.
            if parent is not None:
                row["parent"] = names.get(parent)
            row.pop("id", None)
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
        "  - Each `requirement_groups` row carries its own `options`: the courses that satisfy "
        "that group, each with the catalog's credit figure for it. Every option requires a "
        "minimum grade of C; look its course_code up in `courses` for the title. A group is "
        "referred to by its `name`.",
        "  - Courses in `courses` that are not named in any group's `options` are "
        "ELECTIVES: real courses in this department, not tied to a specific requirement. They "
        "count toward a selective/elective group's credit target same as any listed option.",
        "  - course_aliases.alias_of names an approved substitute: the two codes satisfy the "
        "same requirement and the same prerequisites, so take one, never both.",
        "",
    ]
    for table, source in TABLES.items():
        if table in NESTED_TABLES:
            continue
        rows = _export_rows(db, table, depths)
        parts.append(f"## {table}  ({source}, {len(rows)} rows)")
        parts += [json.dumps(row, sort_keys=True, separators=_ROW_SEPARATORS,
                             ensure_ascii=False) for row in rows]
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
