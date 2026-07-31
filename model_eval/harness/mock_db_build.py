"""Project the plan fixture into the mock database's tables.

THE FIXTURE STAYS THE ONE PLACE TO EDIT. `plan_fixtures/*.yaml` is the scoring authority — a
plan is viable or not according to it and nothing else — so the mock database is DERIVED from
it rather than maintained alongside it. Two hand-maintained copies of the same prerequisite
graph is how you end up scoring a model against an edge it was never shown, which is the one
failure this harness cannot detect from the outside.

`run.py build-mock-db` runs this; `run.py check` re-runs it in memory and fails if the result
differs from what is on disk, so a fixture edit that was never rebuilt cannot reach a run.

WHAT THE PROJECTION ADDS, and why it is not just a reformat: the fixture stores what the SCORER
needs, and the real databases store more than that. Modelling those extra columns honestly is
the point of the exercise —

  * `courses.prerequisites_raw` is NULL, always. Acalog does not publish prerequisites
    (TODO.md §2.5), so the real crawled `courses` row genuinely has nothing in that column, and
    filling it here would misrepresent where the data comes from. The prerequisite text lives
    in `advisor.course_prerequisites`, which is crawled from Banner instead.
  * `course_prerequisites.raw_text` is SYNTHESISED from the structured edges into the phrasing
    Banner uses, so a model sees the messy human sentence next to the parsed form — which is
    what it will see in production, and a fair test of whether the parsed form is the one it
    actually uses.
  * `prereq_groups` is the flat AND-of-ORs shape from 003_course_prerequisites. The fixture's
    edges are all plain ANDs, so every group is a singleton; the nesting is preserved anyway
    because that is the shape the real column has and a model that special-cases singletons
    would break on real data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .fixtures import Fixture
from .mock_db import TABLES

# `refresh-offerings` writes the observation count as a trailing comment on the offered_terms
# line, and YAML drops comments — so it is read back out of the raw text. It is real evidence
# (9 terms is a pattern, 1 is a guess) and dropping it on the way into the mock would make the
# export claim more certainty than the data has.
_OBSERVED_RE = re.compile(
    r'^\s*-\s*code:\s*"([^"]+)"|^\s*offered_terms:\s*\[[^\]]*\]\s*#\s*observed in (\d+)'
)

PROGRAM_ID = "cs-bs-machine-intelligence"


def observed_terms(fixture_path: Path) -> dict[str, int]:
    """course_code -> how many PurdueIO terms its offering pattern was observed across."""
    counts: dict[str, int] = {}
    current: str | None = None
    for line in fixture_path.read_text().splitlines():
        match = _OBSERVED_RE.match(line)
        if not match:
            continue
        if match.group(1):
            current = match.group(1)
        elif current is not None:
            counts[current] = int(match.group(2))
    return counts


def _prereq_sentence(prereqs: list[str], coreqs: list[str]) -> str | None:
    """Banner's own phrasing, rebuilt from the structured edges."""
    parts: list[str] = []
    if prereqs:
        parts.append(
            f"Prerequisite: {' and '.join(prereqs)} with a minimum grade of C."
        )
    if coreqs:
        parts.append(
            f"Corequisite: {' and '.join(coreqs)}. May be taken concurrently."
        )
    return " ".join(parts) or None


def build_tables(fixture: Fixture, *, chain_order: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Every mock table, as rows. ``chain_order`` fixes the order `courses` is written in."""
    by_code = fixture.by_code
    program = {
        "id": PROGRAM_ID,
        "catalog_year": "2026-2027",
        "name": fixture.name,
        "degree_type": "BS",
        "program_type": "Undergraduate concentration",
        "campus": "West Lafayette",
        "total_credits_min": 120,
    }

    courses: list[dict[str, Any]] = []
    prerequisites: list[dict[str, Any]] = []
    offerings: list[dict[str, Any]] = []
    workload: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    counts = observed_terms(fixture.path)

    for code in chain_order:
        course = by_code[code]
        # `prerequisites_raw`, `corequisites_raw` and `description` are omitted rather than
        # written as null on 43 rows: the first two are empty in the REAL table too (see the
        # module docstring), which is a fact the export header states once instead of 43 times.
        courses.append({
            "course_code": code,
            "title": course.title,
            # `credit_hours_max` is dropped: it equals the min for every course in this
            # program (no variable-credit courses), so 43 rows of it restate one sentence the
            # export header can carry. It comes back the moment a variable-credit course lands.
            "credit_hours_min": course.credits,
            "attributes": list(course.requirement_tags),
        })
        # ROWS ONLY FOR COURSES THAT HAVE PREREQUISITES. The real table keeps a
        # `has_prereqs: false` row to distinguish "the page said none" from "never crawled",
        # and that distinction is worth real money on real data — but every course here HAS
        # been crawled, so the export header can state once that an absent course has no
        # prerequisites instead of spending ~13 rows saying it. `has_prereqs` goes with them:
        # empty prereq_groups already says it, and `confidence: "none"` says it a third time.
        if course.prereqs or course.coreqs:
            prerequisites.append({
                "course_code": code,
                "raw_text": _prereq_sentence(list(course.prereqs), list(course.coreqs)),
                # Flat AND-of-ORs: [["CS 25000"], ["CS 25100", "CS 25300"]] reads
                # "CS 25000 AND (CS 25100 OR CS 25300)". Singletons here; see the docstring.
                "prereq_groups": [[p] for p in course.prereqs],
                "coreq_codes": list(course.coreqs),
                "confidence": "high",
            })
        # The real `advisor.course_planner_terms` VIEW, not the base
        # `course_offering_patterns` table: the view is what the planner actually queries, it
        # is the shape the planner wants (a text[] of schedulable terms), and it is shorter
        # than four booleans. Winter is excluded by the view itself; summer appears here
        # whenever the course runs in summers, and whether the planner may USE summer is a
        # separate decision (run.planning_terms), exactly as in production.
        offerings.append({
            "course_code": code,
            "offered_terms": [t for t in ("fall", "spring", "summer")
                              if t in set(course.offered_terms)],
            # 0 = hand-written, never observed. Not the same fact as "observed and absent",
            # and the export prints it so the difference stays visible.
            "terms_observed": counts.get(code, 0),
        })
        # Scale and provenance are stated once in the export header, not on every row.
        workload.append({"course_code": code, "workload_score": course.workload_score})
        if course.equivalent_to:
            aliases.append({"course_code": code, "alias_of": course.equivalent_to,
                            "reason": "approved substitute"})

    groups: list[dict[str, Any]] = []
    options: list[dict[str, Any]] = []
    for order, group in enumerate(fixture.requirement_groups):
        choose = group.get("kind") != "all"
        credits = (float(group.get("choose_credits", 0)) if choose
                   else float(sum(fixture.credits(c) for c in group.get("courses", []))))
        groups.append({
            "id": group["id"],
            "program_id": PROGRAM_ID,
            "parent_group_id": None,
            "name": group["name"],
            "requirement_type": "choose_credits" if choose else "all_of",
            "credits_min": credits,
            "raw_text": (f"Choose {credits:g} credits from the following."
                         if choose else "Take all of the following."),
        })
        for index, code in enumerate(group.get("courses", [])):
            # `course_title` (join to courses), `is_required`/`is_selective_option`
            # (the group's requirement_type says it) and `minimum_grade` (C everywhere, stated
            # once in the export header) are all omitted — 36 rows of restating.
            #
            # `display_order` IS OMITTED TOO, and for a different reason than size. In the real
            # schema it means "the position this option occupies on the catalog page" — but it
            # is the only ordinal in an export whose task is deciding an order, and its values
            # (0..6 within cs-core, 0..8 across the groups) are in exactly the range a
            # semester index would occupy. Asking a 4B model to hold "this integer named
            # `order` is a page layout detail, not the semester to put the course in" is a
            # distinction it does not need to make, since row order already carries it.
            options.append({
                "requirement_group_id": group["id"],
                "course_code": code,
                "credits": float(fixture.credits(code)),
            })

    return {
        "catalog_years": [{
            "label": "2026-2027",
            "start_year": 2026,
            "end_year": 2027,
            "is_archived": False,
        }],
        "programs": [program],
        "requirement_groups": groups,
        "requirement_options": options,
        "courses": courses,
        "course_aliases": aliases,
        "program_notes": [{
            "program_id": PROGRAM_ID,
            "note_type": "concentration",
            "note_text": (
                "Machine Intelligence is a concentration within the Computer Science BS. "
                "Students complete the CS core, then the concentration's required course, one "
                "AI course, and two selectives, alongside the mathematics, laboratory science "
                "and University Core requirements listed in requirement_groups."
            ),
        }],
        "course_prerequisites": prerequisites,
        "course_planner_terms": offerings,
        "course_workload": workload,
    }


def chain_order(fixture: Fixture) -> list[str]:
    """Course codes in prerequisite order: shallowest first, ties broken by code.

    This ordering IS the sequencing hint. It used to be a "[level N]" prefix on every catalog
    line, which a reader could take for a class year — "level 0" reading as "freshman year", a
    constraint the data never expressed. Sorting the rows says the same thing without naming a
    number that invites that reading: a course never appears above something it depends on, so
    reading the table top to bottom is already close to a legal order.
    """
    depth: dict[str, int] = {}
    resolving: set[str] = set()
    by_code = fixture.by_code

    def resolve(code: str) -> int:
        if code in depth:
            return depth[code]
        course = by_code.get(code)
        if course is None or code in resolving:
            return 0
        resolving.add(code)
        value = max((resolve(p) + 1 for p in course.prereqs if p in by_code), default=0)
        resolving.discard(code)
        depth[code] = value
        return value

    for course in fixture.catalog:
        resolve(course.code)
    return sorted(by_code, key=lambda c: (depth.get(c, 0), c))


def render_files(fixture: Fixture) -> dict[str, str]:
    """table -> the exact file contents `build-mock-db` would write. Byte-stable."""
    tables = build_tables(fixture, chain_order=chain_order(fixture))
    missing = set(TABLES) - set(tables)
    if missing:  # pragma: no cover — guards TABLES and the builder drifting apart
        raise ValueError(f"builder produced no rows for declared table(s): {sorted(missing)}")
    return {table: json.dumps(tables[table], indent=2, ensure_ascii=False) + "\n"
            for table in TABLES}


def write_mock_db(fixture: Fixture, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table, text in render_files(fixture).items():
        path = out_dir / f"{table}.json"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def stale_tables(fixture: Fixture, out_dir: Path) -> list[str]:
    """Tables whose file on disk is not what the current fixture would produce."""
    stale = []
    for table, text in render_files(fixture).items():
        path = out_dir / f"{table}.json"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(table)
    return stale
