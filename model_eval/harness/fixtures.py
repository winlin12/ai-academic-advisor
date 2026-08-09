"""Load ``plan_fixtures/*.yaml`` into the harness's planner types.

Two jobs:

1. Turn the fixture's course rows into :class:`planner.Course` objects.
2. Turn its ``requirement_groups`` into the ordered ``remaining_courses`` list a profile
   carries — a port of ``backend/app/services/planner_catalog.select_remaining_courses``
   (required groups first, then just enough selective options to cover each group's credit
   target, counting completed courses toward the group). Mode A has to start from the same
   baseline production would, or it isn't measuring production.

The fixture's raw bytes are hashed into every plan record: edit the fixture and old records
stop being comparable, exactly like the prompt-hash rule.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .planner import Course, Profile

DEFAULT_OPTION_CREDITS = 3.0


@dataclass
class Scenario:
    id: str
    label: str
    profile: Profile
    feedback: str
    assertions: list[dict[str, Any]]
    # True = no legal plan can cover every requirement in the horizon, on purpose. The
    # reachability check in `run.py check` expects the deterministic planner to fail here,
    # and the report scores the row on violations and honest "unplanned" reporting rather
    # than on PLAN_VIABLE (which is 0% for everyone by construction).
    expect_unsatisfiable: bool = False
    # MODE B/C ONLY — never read by the deterministic planner (Mode A), which is why these
    # live on `Scenario` rather than `Profile`: `Profile` mirrors the production app's own
    # student-state type, and neither field has a production counterpart to mirror. See
    # `harness/real_db.build_scenario_database` for how each is applied.
    #
    # Free text ranked against `real_db._GEN_ED_THEMES`, or one of a handful of "no real
    # preference" phrases (also in real_db.py) that resolve to a hand-picked default list —
    # there being no enrollment data anywhere in either crawled database to rank "popular" by.
    # None behaves exactly like "whatever works".
    gen_ed_preference: str | None = None
    # (subject prefix, level) — e.g. ("SPAN", 3). Narrows every world-language menu to that
    # subject's next three courses in sequence and drops every OTHER language's options
    # entirely; see `real_db._language_window`. None leaves language menus in the general
    # gen_ed_preference ranking, same as any other elective group.
    world_language: tuple[str, int] | None = None


@dataclass
class Fixture:
    name: str
    path: Path
    fixture_hash: str
    verified: bool
    catalog: list[Course]
    requirement_groups: list[dict[str, Any]]
    scenarios: list[Scenario]
    # Multi-school support (added 2026-08-08 for the school-sweep feature — see `run.py`'s
    # `discover_fixtures`/`resolve_major`). `slug` is the short name `run.py run`'s default
    # "every school" loop uses for `results/results_<slug>/`; `real_db_program_id` is the real,
    # crawled Postgres program this fixture's scenarios were written against, so Mode B/C read
    # the SAME program Mode A is scored against instead of whatever config.yaml happened to have
    # configured last. Both empty ("") for a fixture that hasn't opted in yet — `run.py` treats
    # an empty `real_db_program_id` as "skip Mode B/C for this fixture" rather than silently
    # falling back to a different program's data.
    slug: str = ""
    real_db_program_id: str = ""
    # "real_db" (default) — Mode B/C's database is fetched live from Postgres via
    # `real_db_program_id`, exactly as every fixture worked before this field existed.
    # "fixture" — Mode B/C's database is built from THIS fixture's own `courses`/
    # `requirement_groups` instead (`real_db.real_db_base_from_fixture`), no Postgres round
    # trip at all. For the school-core/gen-ed+world-language pseudo-major fixtures, whose
    # content has no live-DB equivalent to fetch — see those fixtures' own provenance headers.
    source: str = "real_db"
    # ""/absent (default) = unchanged: every requirement group Postgres returns for this
    # program. "major"/"college"/"university" narrows Mode B/C to just that
    # `real_db._scope_for` tier — see `real_db.fetch_real_db_base`'s `scope_filter` param. A
    # no-op for a `source: fixture` fixture (there is no live DB to filter).
    requirement_scope: str = ""

    @property
    def by_code(self) -> dict[str, Course]:
        return {course.code: course for course in self.catalog}

    def credits(self, code: str) -> int:
        course = self.by_code.get(code)
        return course.credits if course else 0

    @property
    def canonical(self) -> dict[str, str]:
        """code -> the PRIMARY course it counts as. Identity for everything unpaired.

        "MA 16500" -> "MA 16100": an approved substitute satisfies the requirement, and the
        prerequisite, that its primary appears under, and taking both is taking one course
        twice. Collapsing to the primary before any of those three checks is what makes all
        three agree; doing it in only one of them is how you get a plan that covers the degree
        but still reports a prereq violation.

        Chains are followed (A -> B -> C all collapse to C) with a visited set, so a fixture
        typo that makes two courses equivalent to each other cannot hang the scorer.
        """
        direct = {c.code: c.equivalent_to for c in self.catalog if c.equivalent_to}
        out: dict[str, str] = {}
        for course in self.catalog:
            code, seen = course.code, {course.code}
            while direct.get(code) and direct[code] not in seen:
                code = direct[code]
                seen.add(code)
            out[course.code] = code
        return out


def _course(row: dict[str, Any]) -> Course:
    return Course(
        code=row["code"],
        title=row.get("title", ""),
        credits=int(row.get("credits", 0)),
        prereqs=tuple(row.get("prereqs") or ()),
        coreqs=tuple(row.get("coreqs") or ()),
        offered_terms=tuple(row.get("offered_terms") or ()),
        requirement_tags=tuple(row.get("requirement_tags") or ()),
        equivalent_to=str(row.get("equivalent_to") or ""),
    )


def select_remaining_courses(
    groups: list[dict[str, Any]], completed: set[str], credits_of
) -> list[str]:
    """Port of ``planner_catalog.select_remaining_courses`` over fixture groups.

    ``kind: all`` mirrors a required requirement_group; ``kind: choose`` mirrors a selective
    one. Required blocks come before all selectives and duplicates keep their first slot.
    """
    required: list[str] = []
    selective: list[str] = []
    seen: set[str] = set(completed)

    for group in groups:
        if group.get("kind") != "all":
            continue
        for code in group.get("courses", []):
            if code not in seen:
                required.append(code)
                seen.add(code)

    for group in groups:
        if group.get("kind") != "choose":
            continue
        options = list(group.get("courses", []))
        if not options:
            continue
        target = group.get("choose_credits")
        if target is None:
            target = min((credits_of(c) or DEFAULT_OPTION_CREDITS) for c in options)
        needed = float(target) - sum(
            credits_of(code) or DEFAULT_OPTION_CREDITS for code in options if code in completed
        )
        for code in options:
            if needed <= 0:
                break
            if code in seen:
                continue
            selective.append(code)
            seen.add(code)
            needed -= credits_of(code) or DEFAULT_OPTION_CREDITS

    return required + selective


def load_fixture(path: Path) -> Fixture:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    catalog = [_course(row) for row in data["courses"]]
    by_code = {course.code: course for course in catalog}
    groups = data["requirement_groups"]

    def credits_of(code: str) -> int:
        course = by_code.get(code)
        return course.credits if course else 0

    # Major-course load caps: a program-level default every scenario inherits, so the rule is
    # stated once and the eval cannot end up comparing models against different limits by
    # accident. A scenario whose student asked for something lighter overrides it in its own
    # `profile` block — see the mi-* scenarios.
    program = data.get("program") or {}
    default_subject = str(program.get("major_subject", "CS"))
    default_hard = int(program.get("max_major_courses_per_semester", 3))
    default_soft = int(program.get("preferred_major_courses_per_semester", 2))
    # None/absent is meaningful here and must NOT be coerced to an int: it is what tells the
    # planner to derive the target per term rather than pin it.
    default_target = program.get("target_credits_per_semester")
    # The registrar's ceiling, program-level for the same reason the major cap is: it is a rule
    # about the university, not about any one student, so no scenario should be able to drift
    # off it by accident. A scenario may still override it (a student on academic probation has
    # a genuinely lower one), but none does today.
    default_hard_credits = int(program.get("hard_credit_cap", 18))

    scenarios: list[Scenario] = []
    for row in data.get("scenarios", []):
        p = row["profile"]
        completed = list(p.get("completed_courses") or [])
        remaining = select_remaining_courses(groups, set(completed), credits_of)
        scenarios.append(
            Scenario(
                id=row["id"],
                label=row.get("label", row["id"]),
                profile=Profile(
                    name=p.get("name", "Student"),
                    degree_program=p.get("degree_program", ""),
                    completed_courses=completed,
                    remaining_courses=remaining,
                    start_term=p.get("start_term", "fall"),
                    start_year=int(p.get("start_year", 2026)),
                    semesters_to_plan=int(p.get("semesters_to_plan", 8)),
                    max_credits_per_semester=int(p.get("max_credits_per_semester", 16)),
                    hard_credit_cap=int(p.get("hard_credit_cap", default_hard_credits)),
                    target_credits_per_semester=(
                        int(t) if (t := p.get("target_credits_per_semester",
                                              default_target)) is not None else None),
                    major_subject=str(p.get("major_subject", default_subject)),
                    max_major_courses_per_semester=int(
                        p.get("max_major_courses_per_semester", default_hard)),
                    preferred_major_courses_per_semester=int(
                        p.get("preferred_major_courses_per_semester", default_soft)),
                ),
                feedback=(row.get("feedback") or "").strip(),
                assertions=list(row.get("assertions") or []),
                expect_unsatisfiable=bool(row.get("expect_unsatisfiable", False)),
                gen_ed_preference=row.get("gen_ed_preference"),
                world_language=(
                    (str(row["world_language"]["subject"]).upper(),
                     int(row["world_language"]["level"]))
                    if row.get("world_language") else None
                ),
            )
        )

    return Fixture(
        name=data["program"]["name"],
        path=Path(path),
        fixture_hash=hashlib.sha256(raw).hexdigest()[:16],
        verified=bool(data["program"].get("verified", False)),
        catalog=catalog,
        requirement_groups=groups,
        scenarios=scenarios,
        slug=str(program.get("slug") or Path(path).stem),
        real_db_program_id=str(program.get("real_db_program_id") or ""),
        source=str(program.get("source") or "real_db"),
        requirement_scope=str(program.get("requirement_scope") or ""),
    )
