"""Scores a Mode B plan against the REAL requirement structure and REAL course facts of
whichever program `real_db.py` actually loaded — not a hand-authored fixture.

WHY THIS EXISTS, AND WHY IT IS NOW THE ONLY SCORER MODE B USES. `plan_scorers.score_plan()` is
fixture-only: its course universe, credits, prereqs, coreqs, term offerings and requirement
groups all come from `plan_fixtures/cs_machine_intelligence.yaml` regardless of which program
the model was actually SHOWN — and `run.py --major` (or a `real_db.program_id` that drifts from
the fixture's own program) can point Mode B at any program in the catalog. Two ways that showed
up as a correctness bug, not just a display quirk:

  * A model choosing a real, valid course outside the fixture's course list — the WGSS incident
    (`real_db.py`'s module docstring) got scored "hallucinated" because the CS fixture's course
    universe had never heard of the course, even though it was right there in `courses` in the
    same prompt. Confirmed again 2026-08-04: a Mechanical Engineering run (real_db.program_id
    pointed off the fixture's own program) flagged 24 genuinely real ME courses as
    `hallucinated_course`, and a same-program CS run still showed `gen-ed-core`/`science` as
    "missing" because the FIXTURE invented specific gen-ed course codes (SCLA 10100, PHYS
    17200, ...) that were never real, enumerable requirements — the real catalog correctly
    states these in prose (`unresolved_requirement_groups`), and the fixture's guess at turning
    prose into a course list is exactly the kind of hardcoding this module replaces.
  * `missing_requirements` read back labels (`cs-elective`, `gen-ed-selective`) that describe
    the FIXTURE's invented taxonomy, not the real program's actual `requirement_groups` names —
    misleading regardless of which program real_db.program_id names.

So `plan_scorers.score_plan` is fixture-only BY DESIGN and stays exactly that for Mode A (the
deterministic planner's own output, "legal by construction" against the SAME fixture it was
built from — nothing about a real database belongs there) — see `runner.run_mode_a`. Mode B is
the one call site that grammar-constrains the model to `ctx.database.course_codes` and shows it
`ctx.database`'s tables as the prompt, so Mode B is the one call site this module scores
instead, not alongside: `runner._run_freeform_plan` no longer calls `plan_scorers.score_plan`
for Mode B at all.

FIELD-COMPATIBLE WITH `plan_scorers.PlanScore` ON PURPOSE. `RealScore` carries the same
attribute names (`viable`, `violations`, `semester_credits`, `major_courses_per_semester`, ...)
so `runner._score_detail` and `plan_scorers.check_assertions` — both written against
`PlanScore` — work on a `RealScore` UNCHANGED; duck typing, not a shared base class, because a
shared base class would tempt someone into making the two interchangeable in the other
direction too, and Mode A must never silently start reading real-DB data it was never shown.
`groups`/`unresolved` are the one real addition: per-group detail a flat `PlanScore` has no
slot for, used by `runner._real_score_detail` for the "every checkable requirement group" table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog_export import CatalogDatabase
from .planner import Profile, first_planning_term, is_major_course, next_term
from .plan_scorers import normalize_code


@dataclass
class RealGroupResult:
    name: str
    selective: bool
    credits_required: float | None
    credits_filled: float
    satisfied: bool
    filled: list[str] = field(default_factory=list)
    missing_label: str | None = None


@dataclass
class RealUnresolved:
    name: str
    raw_text: str | None
    scope: str = "program"


@dataclass
class RealScore:
    """Everything `plan_scorers.PlanScore` tracks, computed from the real database instead of
    the fixture, plus per-group detail (`groups`, `unresolved`) the flat original has no slot
    for. See the module docstring for why the field names match `PlanScore` exactly."""

    structure_ok: bool = True
    viable: bool = False
    violations: list[str] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    # Over CHECKABLE groups only (`groups`, below) — `unresolved` cannot be decided either way,
    # and counting it against coverage would tell a student they are behind when the honest
    # answer is "we do not know". See `UnresolvedRequirement`'s docstring in the app for the
    # same call made the same way in production.
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = field(default_factory=list)
    planned_credits: int = 0
    semester_credits: list[int] = field(default_factory=list)
    credit_spread: int = 0
    idle_credits: int = 0
    semesters_used: int = 0
    semesters_available: int = 0
    major_courses_per_semester: list[int] = field(default_factory=list)
    soft_major_overloads: int = 0
    soft_credit_overages: int = 0
    hallucinated: list[str] = field(default_factory=list)
    assertions_passed: dict[str, bool] = field(default_factory=dict)

    # Real-DB-specific detail, absent from `PlanScore`.
    groups: list[RealGroupResult] = field(default_factory=list)
    unresolved: list[RealUnresolved] = field(default_factory=list)

    @property
    def groups_satisfied(self) -> int:
        return sum(1 for g in self.groups if g.satisfied)

    @property
    def groups_total(self) -> int:
        return len(self.groups)

    @property
    def coverage(self) -> float:
        """Alias for `requirement_coverage` — kept for `runner._real_score_detail`, which reads
        it by this name."""
        return self.requirement_coverage

    def as_record(self) -> dict[str, Any]:
        """`PlanScore.as_record()`'s exact key surface, so `rec.update(real_score.as_record())`
        in `runner._run_freeform_plan` IS Mode B's primary, unprefixed score — no translation
        layer, and report.py needs no changes to read it. `real_groups`/`real_unresolved` are
        additive: the per-group breakdown `PlanScore` cannot express, kept under a distinct
        name so it never collides with anything `plan_scorers` writes."""
        return {
            "structure_ok": self.structure_ok,
            "plan_viable": self.viable,
            "violations": self.violations,
            "violation_counts": self.violation_counts,
            "requirement_coverage": round(self.requirement_coverage, 4),
            "missing_requirements": self.missing_requirements,
            "planned_credits": self.planned_credits,
            "semester_credits": self.semester_credits,
            "credit_spread": self.credit_spread,
            "idle_credits": self.idle_credits,
            "semesters_used": self.semesters_used,
            "semesters_available": self.semesters_available,
            "major_courses_per_semester": self.major_courses_per_semester,
            "soft_major_overloads": self.soft_major_overloads,
            "soft_credit_overages": self.soft_credit_overages,
            "hallucinated_courses": self.hallucinated,
            "assertions_passed": self.assertions_passed,
            "real_groups": [
                {
                    "name": g.name, "selective": g.selective,
                    "credits_required": g.credits_required, "credits_filled": g.credits_filled,
                    "satisfied": g.satisfied, "filled": g.filled, "missing": g.missing_label,
                }
                for g in self.groups
            ],
            "real_unresolved": [
                {"name": u.name, "raw_text": u.raw_text, "scope": u.scope}
                for u in self.unresolved
            ],
        }


def _terms(profile: Profile, count: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(count):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def _groups_from_tables(database: CatalogDatabase) -> list[dict[str, Any]]:
    """Reconstruct each requirement group's courses and selective/take-all flag from
    `requirement_groups`/`requirement_options` — the same two tables `catalog_export
    .render_context` prints, so this scores exactly what the model was shown, nothing else."""
    options_by_group: dict[str, list[str]] = {}
    for row in database.rows("requirement_options"):
        options_by_group.setdefault(row["requirement_group_id"], []).append(row["course_code"])
    return [
        {
            "id": row["id"], "name": row["name"],
            "selective": row["requirement_type"] == "choose_credits",
            "credits_min": row["credits_min"],
            "courses": options_by_group.get(row["id"], []),
        }
        for row in database.rows("requirement_groups")
        if options_by_group.get(row["id"])
    ]


def score_against_real_db(
    database: CatalogDatabase, profile: Profile, semesters: list[list[str]],
    *, assertions: list[dict[str, Any]] | None = None,
) -> RealScore:
    """Score a schedule (list of per-semester course-code lists) against the requirement
    groups and course facts actually loaded for `database`'s program — every check
    `plan_scorers.score_plan` runs, reimplemented against real rows instead of the fixture. See
    the module docstring for why Mode B is scored by this and only this.
    """
    by_code = {row["course_code"]: row for row in database.rows("courses")}
    alias_of = {row["course_code"]: row["alias_of"] for row in database.rows("course_aliases")}
    canon = lambda code: alias_of.get(code, code)  # noqa: E731

    score = RealScore()
    counts: dict[str, int] = {}

    def flag(kind: str, detail: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        score.violations.append(f"{kind}: {detail}")

    completed = {canon(normalize_code(c)) for c in profile.completed_courses}
    seen: set[str] = set()
    terms = _terms(profile, len(semesters))

    soft_major_cap = min(profile.preferred_major_courses_per_semester,
                         profile.max_major_courses_per_semester)

    for index, (raw_codes, (term, year)) in enumerate(zip(semesters, terms)):
        codes = [normalize_code(c) for c in raw_codes]
        this_semester = {canon(c) for c in codes}
        credits = 0
        major_courses = 0
        # Prereqs/coreqs evaluated against courses completed BEFORE this semester, same rule
        # `plan_scorers.score_plan` uses — a course taken alongside its own prerequisite is a
        # scheduling error, not a corequisite.
        available = set(completed)

        for code in codes:
            course = by_code.get(code)
            if course is None:
                score.hallucinated.append(code)
                flag("hallucinated_course", f"{code} (semester {index + 1}) is not in the "
                                            f"real catalog export the model was shown")
                continue
            key = canon(code)
            if key in seen or key in completed:
                # Every real caller runs `plan_scorers.dedupe_semesters` before scoring, so
                # this should never fire in practice — kept as a defensive SKIP, not a
                # violation, matching `plan_scorers.score_plan`'s identical change. See
                # `plan_scorers`'s module docstring for why duplicates left scoring entirely.
                continue
            seen.add(key)

            credits += int(course.get("credit_hours_min") or 0)
            if is_major_course(code, profile.major_subject):
                major_courses += 1

            missing_groups = [
                " or ".join(group) for group in (course.get("prereq_groups") or [])
                if not any(canon(normalize_code(c)) in available for c in group)
            ]
            if missing_groups:
                flag("prereq_violation",
                     f"{code} in semester {index + 1} needs {', '.join(missing_groups)} first")

            # Looser than a prerequisite the same way `plan_scorers.score_plan` reads it: a
            # coreq taken IN this semester or earlier both count. See that function's comment
            # for the CS 37300/STAT 35000 incident this distinction fixes.
            missing_co = [
                c for c in (course.get("coreq_codes") or [])
                if canon(normalize_code(c)) not in available | this_semester
            ]
            if missing_co:
                flag("coreq_violation",
                     f"{code} in semester {index + 1} needs {', '.join(missing_co)} "
                     f"in the same semester or earlier")

        score.semester_credits.append(credits)
        score.major_courses_per_semester.append(major_courses)
        # BOTH SOFT ONLY, never gate `viable` — a heavy semester, credit-wise or major-course-
        # wise, is the student's own call to adjust and can change from one advising
        # conversation to the next, unlike a missed prerequisite. See `plan_scorers.py`'s
        # module docstring for the same call made the same way for Mode A/C.
        if credits > profile.max_credits_per_semester:
            score.soft_credit_overages += 1
        if major_courses > soft_major_cap:
            score.soft_major_overloads += 1

        completed |= {canon(code) for code in codes if code in by_code}

    score.violation_counts = counts
    score.planned_credits = sum(score.semester_credits)
    score.semesters_available = profile.semesters_to_plan
    score.semesters_used = sum(1 for c in score.semester_credits if c > 0)
    nonempty = [c for c in score.semester_credits if c > 0]
    score.credit_spread = (max(nonempty) - min(nonempty)) if nonempty else 0

    canonical_satisfied = {canon(normalize_code(c)) for c in profile.completed_courses} | seen
    missing_labels: list[str] = []
    for group in _groups_from_tables(database):
        courses = group["courses"]
        if group["selective"]:
            filled = [c for c in courses if canon(c) in canonical_satisfied]
            credits_filled = sum(float(by_code[c].get("credit_hours_min") or 0)
                                 for c in filled if c in by_code)
            target = float(group["credits_min"] or 0) or 3.0
            satisfied = credits_filled >= target
            missing_label = (None if satisfied else
                             f"{target - credits_filled:g} more credit(s) from this list")
            score.groups.append(RealGroupResult(
                name=group["name"], selective=True, credits_required=group["credits_min"],
                credits_filled=credits_filled, satisfied=satisfied, filled=filled,
                missing_label=missing_label,
            ))
        else:
            missing = [c for c in courses if canon(c) not in canonical_satisfied]
            filled = [c for c in courses if c not in missing]
            satisfied = not missing
            missing_label = None if satisfied else f"missing {', '.join(missing)}"
            score.groups.append(RealGroupResult(
                name=group["name"], selective=False, credits_required=group["credits_min"],
                credits_filled=float(len(filled)), satisfied=satisfied, filled=filled,
                missing_label=missing_label,
            ))
        if missing_label:
            missing_labels.append(f"{group['name']}: {missing_label}")

    score.requirement_coverage = score.groups_satisfied / score.groups_total \
        if score.groups_total else 1.0
    score.missing_requirements = missing_labels
    score.viable = not counts and score.requirement_coverage >= 1.0

    # Same idle-capacity diagnostic as `plan_scorers.score_plan`: credits the student could
    # legally have taken but was not given, counted only while checkable requirements are still
    # unmet. Zero when coverage is complete.
    if score.requirement_coverage < 1.0:
        cap = profile.max_credits_per_semester
        horizon = max(profile.semesters_to_plan, len(score.semester_credits))
        padded = score.semester_credits + [0] * (horizon - len(score.semester_credits))
        score.idle_credits = sum(max(cap - c, 0) for c in padded[:horizon])

    score.unresolved = [
        RealUnresolved(name=row["name"], raw_text=row.get("raw_text"),
                       scope=row.get("scope", "program"))
        for row in database.rows("unresolved_requirement_groups")
    ]

    if assertions:
        # Duck-typed: `check_assertions` only reads `.semester_credits` and
        # `.major_courses_per_semester` off `score`, both present here under the same names.
        from .plan_scorers import check_assertions
        score.assertions_passed = check_assertions(
            assertions, semesters, score, canonical={code: canon(code) for code in by_code})

    return score
