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


def _course(row: dict[str, Any]) -> Course:
    return Course(
        code=row["code"],
        title=row.get("title", ""),
        credits=int(row.get("credits", 0)),
        prereqs=tuple(row.get("prereqs") or ()),
        offered_terms=tuple(row.get("offered_terms") or ()),
        requirement_tags=tuple(row.get("requirement_tags") or ()),
        workload_score=int(row.get("workload_score", 3)),
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
