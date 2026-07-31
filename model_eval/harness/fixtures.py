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

A THIRD, SEPARATE THING lives here too: :func:`load_sample_plan`, which reads the department's
published sample four-year plan for MODE C. It is loaded from its own file and is NOT folded
into ``fixture_hash``, because it is prompt input rather than scoring authority — it cannot
change how any plan is scored, so adding it must not invalidate the Mode A/B records already
on disk. It travels in Mode C's STATIC PROMPT hash instead, which is where a change to it
genuinely does invalidate comparisons.
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


@dataclass
class Fixture:
    name: str
    path: Path
    fixture_hash: str
    verified: bool
    catalog: list[Course]
    requirement_groups: list[dict[str, Any]]
    scenarios: list[Scenario]

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


@dataclass
class SamplePlanEntry:
    """One row of the published sample plan, as the catalog prints it."""

    text: str
    # The scoring fixture's course this row places, if any. Placeholder rows
    # ("Elective", "Science Core Selection") have none — the catalog does not resolve them.
    code: str | None = None
    # Codes the row names that are NOT in the scoring fixture's catalog: "or MA 16500",
    # "(CS 19300 suggested)". Kept so a copied-from-the-template hallucination is
    # distinguishable from a free-hand one.
    also_cites: tuple[str, ...] = ()


@dataclass
class SamplePlanSemester:
    label: str
    term: str
    credits: str
    entries: list[SamplePlanEntry]

    @property
    def codes(self) -> list[str]:
        return [e.code for e in self.entries if e.code]


@dataclass
class SamplePlan:
    """The department's published sample four-year plan. MODE C's extra input.

    ``plan_hash`` is over the file's raw bytes and goes into Mode C's static prompt hash, so
    editing the sample plan invalidates Mode C comparisons exactly the way editing a system
    prompt does — and leaves Mode A/B alone, which is correct, since they never saw it.
    """

    path: Path
    plan_hash: str
    source: dict[str, Any]
    semesters: list[SamplePlanSemester]

    @property
    def slot_of(self) -> dict[str, int]:
        """code -> the 0-based semester index the sample plan puts it in (first wins)."""
        out: dict[str, int] = {}
        for index, semester in enumerate(self.semesters):
            for code in semester.codes:
                out.setdefault(code, index)
        return out

    @property
    def off_catalog_codes(self) -> set[str]:
        """Every code the plan names that the scoring fixture's catalog does not contain."""
        return {c for s in self.semesters for e in s.entries for c in e.also_cites}


def load_sample_plan(path: Path) -> SamplePlan:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    semesters = [
        SamplePlanSemester(
            label=str(row.get("label", f"Semester {i + 1}")),
            term=str(row.get("term", "")).lower(),
            credits=str(row.get("credits", "")),
            entries=[
                SamplePlanEntry(
                    text=str(entry["text"]),
                    code=entry.get("code"),
                    also_cites=tuple(entry.get("also_cites") or ()),
                )
                for entry in row.get("entries") or []
            ],
        )
        for i, row in enumerate(data.get("semesters") or [])
    ]
    return SamplePlan(
        path=Path(path),
        plan_hash=hashlib.sha256(raw).hexdigest()[:16],
        source=data.get("source") or {},
        semesters=semesters,
    )


def _course(row: dict[str, Any]) -> Course:
    return Course(
        code=row["code"],
        title=row.get("title", ""),
        credits=int(row.get("credits", 0)),
        prereqs=tuple(row.get("prereqs") or ()),
        coreqs=tuple(row.get("coreqs") or ()),
        offered_terms=tuple(row.get("offered_terms") or ()),
        requirement_tags=tuple(row.get("requirement_tags") or ()),
        workload_score=int(row.get("workload_score", 3)),
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
    )
