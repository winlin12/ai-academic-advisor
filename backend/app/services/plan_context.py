"""The database export Mode B plans from — the model's entire view of the catalog.

PORT OF ``model_eval/harness/mock_db.render_context``, reading real Postgres rows instead of
the mock's JSON files. Keeping the two byte-comparable is the whole point: the eval's Mode B
numbers for Gemma 4 26B describe a model reading THIS document, and a serving path that
reformatted it would be running a prompt nobody measured. The harness's own header explains
why the format is what it is (JSON per row so every value carries its column name; row order
is prerequisite-chain order so reading top to bottom is close to a legal scheduling order);
those decisions are reproduced here rather than re-derived.

WHAT IS DIFFERENT FROM THE MOCK, and it is the reason this file exists at all:

  * The mock exports one program because it only has one. Real Postgres holds 951, and
    ``render_program_context`` is scoped to the one the student picked — the "no retrieval
    yet, on purpose" note in mock_db.py said the whole-database export was "only tenable
    because a single-program mock is a few thousand tokens". Per-program scoping is the
    cheapest retrieval step that keeps that true.
  * It is BUDGETED. A crawled program can list far more options than the fixture does, and
    llama-server's context is fixed at launch — an overflowing prompt is a failed request, not
    a slow one. ``context_budget_tokens`` trims the tables that can grow without bound
    (selective menus) and says in the document that it did, so the model never silently
    believes a truncated menu is the whole one.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.services.planner_db import ProgramCatalog

# Rows are printed one per line, compact. A pretty-printed export of the same rows is ~2.5x the
# tokens for no added information. Every value still carries its column name, which is the
# property that actually helps a small model. ensure_ascii=False because program names carry
# em-dashes and — is three tokens of nothing.
_SEPARATORS = (",", ":")

_SEASONS = ("fall", "spring", "summer")

# Same columns the mock declares it drops, for the same reasons. Stated in the export so the
# omission is visible to the reader rather than looking like the export is the whole truth.
OMITTED_COLUMNS = (
    "uuid primary keys and foreign keys (natural keys — course_code, program name — are used "
    "instead); provenance columns (source_page_id, sync_run_id, updated_at, term_code); "
    "columns whose value is derivable from course_code (subject, number); per-season "
    "observation counts (only the total, terms_observed, is kept); courses.description, which "
    "holds Purdue's catalog prose; and the published sample 4-year plan, which is a schedule "
    "rather than a requirement"
)


# CHARS PER TOKEN, and 4 is the wrong number here. The 4:1 rule of thumb is for English prose;
# this document is dense JSON — `{"course_code":"CS 25100","credit_hours_min":3,...}` — where
# punctuation, quoted keys and course codes all tokenize short. MEASURED against the served
# model on the Machine Intelligence export: 32,416 characters became 14,445 tokens, i.e. 2.24
# chars/token, so the 4:1 estimate under-counted the prompt by 44%. That is not a rounding
# error: the budget check passed, the real prompt filled 14.4k of a 16,384-token window, the
# 2,048-token plan had ~1.9k of room, and every draft hit the ceiling mid-JSON and failed to
# parse. The whole feature was down and the estimate is why it looked fine.
#
# 2.2 is deliberately BELOW the measured ratio, because over-estimating the prompt costs some
# unused window and under-estimating costs every plan.
_CHARS_PER_TOKEN = 2.2


def approx_tokens(text: str) -> int:
    """Conservative token estimate for the export. See ``_CHARS_PER_TOKEN``.

    An estimate rather than a tokenizer round-trip on purpose: asking llama-server to count
    would put a network call on the path that builds the prompt, and make plan generation fail
    differently when the model is down than when it is up.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _row(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=_SEPARATORS, ensure_ascii=False)


def _chain_depth(catalog: ProgramCatalog) -> dict[str, int]:
    """Longest prerequisite chain behind each course. Computed, never stored.

    ``planner_db`` already sorts ``catalog.courses`` by this, so recomputing it here is only
    to PRINT it — the export states the number because the model is told what it is not (a
    year, a class standing, a semester index).
    """
    known = {course.code for course in catalog.courses}
    edges = {course.code: [c for group in course.prereq_groups for c in group if c in known]
             for course in catalog.courses}
    depth: dict[str, int] = {}
    resolving: set[str] = set()

    def resolve(code: str) -> int:
        if code in depth:
            return depth[code]
        if code in resolving:
            return 0
        resolving.add(code)
        value = max((resolve(edge) + 1 for edge in edges.get(code, [])), default=0)
        resolving.discard(code)
        depth[code] = value
        return value

    return {code: resolve(code) for code in known}


_OFFERING_HEADLINE = (
    "NOT EVERY COURSE RUNS EVERY TERM. Most rows below list all three terms, but the ones that "
    "do not are hard constraints: a course placed in a term outside its `offered_terms` is a "
    "course the student cannot register for, and no later semester fixes it. Check this list "
    "BEFORE placing a course, not after:"
)


def offering_note(rows: list[dict[str, Any]]) -> list[str]:
    """Name the term-restricted courses ABOVE the rows, instead of only inside them.

    Straight port of ``mock_db._offering_note``, and it earns its place the same way: this is
    a presentation change only — every fact below is read off the ``offered_terms`` values
    printed underneath it — but it is the difference between a constraint the model can notice
    and one buried as a single field on 56 near-identical rows, most of which say "any term is
    fine". ``term_offering_violation`` was firing on the same two fall-only courses plan after
    plan until the eval added this.

    DERIVED, never hand-written. A hand-written offering note is how the eval's fixture came to
    record CS 47300 as spring-only when it runs in falls, and a prompt that disagrees with its
    own data is worse than a quiet one.
    """
    by_terms: dict[tuple[str, ...], list[str]] = {}
    for row in rows:
        by_terms.setdefault(tuple(row.get("offered_terms") or ()), []).append(row["course_code"])

    restricted = {terms: codes for terms, codes in by_terms.items()
                  if set(terms) != set(_SEASONS)}
    if not restricted:
        return []

    note = [_OFFERING_HEADLINE]
    # Fewest terms first — the tightest restrictions break the most plans, so they read first.
    for terms in sorted(restricted, key=lambda t: (len(t), _SEASONS.index(t[0]) if t else -1)):
        codes = ", ".join(restricted[terms])
        missing = "/".join(s for s in _SEASONS if s not in terms)
        if not terms:
            note.append(f"  - NEVER OFFERED, do not schedule at all: {codes}")
        elif len(terms) == 1:
            note.append(f"  - {terms[0].upper()} ONLY — never {missing}: {codes}")
        else:
            note.append(f"  - {'/'.join(terms)} only, never {missing}: {codes}")
    note.append("Every course not named above runs in all three terms.")
    return note


_SOURCE_NOTE = (
    "  - advisor           : what that catalog does not publish. `course_planner_terms` is "
    "OBSERVED — a course is offered in fall because it HAS RUN in falls, and `terms_observed` "
    "is how many terms of history that is based on (0 = never observed, the terms are an "
    "editorial guess). `course_prerequisites` is read from course-detail pages and carries a "
    "`confidence`."
)

# The column glossary. Kept as one tuple of finished strings rather than as a list of
# implicitly-concatenated fragments so it is obvious where each bullet ends — a missing comma
# in the old shape silently glued two rules into one and the model would never have told us.
_COLUMN_NOTES = (
    ("  - A course with NO ROW in `course_prerequisites` has no prerequisites and no "
     "corequisites. Absence is a fact, not a gap."),

    "  - Every course is fixed-credit: `credit_hours_min` is the credit value.",

    ('  - course_prerequisites.prereq_groups is AND-of-ORs: [["CS 25000"],["CS 25100",'
     '"CS 25300"]] means CS 25000 AND (CS 25100 OR CS 25300). [] means no prerequisites.'),

    "  - course_prerequisites.coreq_codes may be taken in the SAME semester or earlier.",

    ("  - course_prerequisites.chain_depth is COMPUTED, not stored: how many courses deep the "
     "longest prerequisite chain behind this one runs. It is NOT a year, a class standing, or a "
     "semester number — chain_depth 3 does not mean 'junior year' or 'semester 3', it means "
     "three courses must be completed in sequence first, which can happen at any point in the "
     "plan. `courses` is listed in this order, so a course never appears above one it depends on."),

    ('  - requirement_options.alternatives lists interchangeable courses: "MA 26100 or '
     'MA 27101" is ONE requirement. Take exactly one of them, never both.'),

    ("  - Every requirement_options row requires a minimum grade of C; join it to `courses` on "
     "course_code for the title and credits."),

    ("  - course_aliases.alias_of names an approved substitute: the two codes satisfy the same "
     "requirement and the same prerequisites, so take one, never both."),

)


def _header(catalog: ProgramCatalog) -> list[str]:
    return [
        "DATABASE EXPORT — advisor catalog snapshot, read-only. One JSON row per line.",
        "",
        (f"This is the complete contents of the advisor's catalog database for {catalog.name} "
         f"({catalog.catalog_year}). Every course that exists for this program is in "
         f"`courses`; nothing outside this export exists."),
        "",
        "Sources, which differ in reliability:",
        "  - catalog_ingestion : crawled from Purdue's official course catalog.",
        _SOURCE_NOTE,
        "",
        f"Not in this export: {OMITTED_COLUMNS}.",
        "",
        "How to read the less obvious columns:",
        *_COLUMN_NOTES,
        "",
    ]


def render_program_context(
    catalog: ProgramCatalog, *, budget_tokens: int | None = None
) -> str:
    """The program's catalog, as the model sees it. Deterministic for a given database state.

    ``budget_tokens`` caps the whole document. Requirement OPTION rows are what get trimmed
    when it binds, because they are the only table that grows without bound (a selective menu
    can list forty courses where the degree needs two) and the only one whose tail a planner
    can lose without losing a hard constraint — a prerequisite edge or a term restriction it
    never saw is a plan that is simply wrong.
    """
    budget = budget_tokens or settings.plan_context_token_budget
    depths = _chain_depth(catalog)
    by_code = catalog.by_code

    parts: list[str] = _header(catalog)

    parts.append("## programs  (catalog_ingestion, 1 row)")
    parts.append(_row({
        "name": catalog.name,
        "catalog_year": catalog.catalog_year,
        "degree_type": catalog.degree_type,
        "program_type": catalog.program_type,
        "campus": catalog.campus,
        "total_credits_min": catalog.total_credits_min,
    }))
    parts.append("")

    if catalog.notes:
        parts.append(f"## program_notes  (catalog_ingestion, {len(catalog.notes)} rows)")
        parts += [_row({"note_text": note}) for note in catalog.notes]
        parts.append("")

    parts.append(f"## requirement_groups  (catalog_ingestion, {len(catalog.groups)} rows)")
    for group in catalog.groups:
        parts.append(_row({
            "id": group.id,
            "name": group.name,
            # The vocabulary PLAN_SYSTEM's "where things are" section names, not the crawler's
            # raw requirement_type — the model is told what `all_of` and `choose_credits` mean
            # and nothing about `foundation` or `college`.
            "requirement_type": "choose_credits" if group.selective else "all_of",
            "credits_min": group.credits_min,
        }))
    parts.append("")

    # Options carry their alternatives inline: one row per requirement, not one per course, so
    # "MA 26100 or MA 27101" cannot read as two separate things to schedule.
    option_rows: list[tuple[str, dict[str, Any]]] = []
    for group in catalog.groups:
        for run in group.options:
            payload: dict[str, Any] = {
                "requirement_group_id": group.id,
                "course_code": run[0],
                "credits": catalog.credits(run[0]),
            }
            if len(run) > 1:
                payload["alternatives"] = run[1:]
            option_rows.append((group.id, payload))

    # Everything except the option table is a hard constraint and is never trimmed; the option
    # table gets whatever budget is left.
    fixed = list(parts)
    course_rows = [
        {
            "course_code": course.code,
            "title": course.title,
            "credit_hours_min": course.credits,
            "attributes": course.requirement_tags,
        }
        for course in catalog.courses
    ]
    prereq_rows = [
        {
            "course_code": code,
            "raw_text": row.get("raw_text"),
            "prereq_groups": row.get("prereq_groups") or [],
            "coreq_codes": row.get("coreq_codes") or [],
            "confidence": row.get("confidence"),
            "chain_depth": depths.get(code, 0),
        }
        for code in (c.code for c in catalog.courses)
        if (row := catalog.prereq_rows.get(code)) is not None
    ]
    planner_term_rows = [
        {
            "course_code": course.code,
            "offered_terms": list(course.offered_terms),
            "terms_observed": int(
                (catalog.offering_rows.get(course.code) or {}).get("terms_observed") or 0
            ),
        }
        for course in catalog.courses
    ]
    alias_rows = [
        {"course_code": substitute, "alias_of": primary, "reason": "approved substitute"}
        for substitute, primary in sorted(catalog.aliases.items())
        if primary in by_code
    ]

    tail: list[str] = []
    tail.append(f"## courses  (catalog_ingestion, {len(course_rows)} rows)")
    tail.append("Listed in prerequisite order — a course never appears above one it needs.")
    tail += [_row(row) for row in course_rows]
    tail.append("")

    tail.append(f"## course_prerequisites  (advisor, {len(prereq_rows)} rows)")
    tail += [_row(row) for row in prereq_rows]
    tail.append("")

    tail.append(f"## course_planner_terms  (advisor, {len(planner_term_rows)} rows)")
    tail += offering_note(planner_term_rows)
    tail += [_row(row) for row in planner_term_rows]
    tail.append("")

    if alias_rows:
        tail.append(f"## course_aliases  (catalog_ingestion, {len(alias_rows)} rows)")
        tail += [_row(row) for row in alias_rows]
        tail.append("")

    spent = approx_tokens("\n".join(fixed + tail))
    remaining = budget - spent

    kept: list[str] = []
    dropped = 0
    for _group_id, payload in option_rows:
        line = _row(payload)
        cost = approx_tokens(line)
        if cost > remaining:
            dropped += 1
            continue
        kept.append(line)
        remaining -= cost

    options_block = [f"## requirement_options  (catalog_ingestion, {len(option_rows)} rows)"]
    if dropped:
        # Said out loud, because a model that believes a trimmed menu is the whole menu will
        # confidently report a requirement as uncoverable. Better it knows the list is partial.
        options_block.append(
            f"NOTE: {dropped} of these rows were omitted to fit the context window. Every "
            f"requirement group above is still listed; some selective groups have more options "
            f"than are shown, so treat a group's visible options as a sample, not the whole menu."
        )
    options_block += kept
    options_block.append("")

    return "\n".join(fixed + options_block + tail).rstrip() + "\n"
