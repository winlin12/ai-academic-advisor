"""Vendored copy of the app's deterministic planner — stdlib only.

WHY A COPY. The harness's standing rule is that it depends on nothing in the app (no
pydantic, no FastAPI, no database), so it can run on a bare Python install on whichever
box has the GPU. But Mode A of the plan eval measures the app's REAL revise-plan path,
which means the model's ``PlanEditProposal`` has to be folded into a profile and re-planned
by the same algorithm production uses. Reimplementing it "roughly" would make Mode A
measure the harness instead of the app.

So this file is a deliberate, faithful port of:

    backend/app/services/planner.py            -> generate_plan / prereqs_satisfied / next_term
    backend/app/services/advisor_agent.py      -> _apply_proposal / _severity

DRIFT IS THE COST. If someone changes the app's planner and not this file, Mode A silently
stops measuring production. ``python run.py parity`` diffs the two implementations on a
generated battery of profiles whenever the backend happens to be importable, and the report
prints whether that check has been run. Keep the two in sync or the Mode A column is a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Mirrors backend/app/services/planner.py. Summer is excluded from PLANNING by default
# (backend setting ``planner_include_summer``) — it is a real term with real offerings, but
# summer enrolment carries cost/aid/residency consequences a planner cannot reason about.
# The harness sets this from config.yaml's ``run.planning_terms`` at load time so the eval and
# the app never silently disagree about what a valid semester is.
ALL_TERMS = ["fall", "spring", "summer"]
TERM_ORDER = ["fall", "spring"]


def set_planning_terms(terms: list[str]) -> None:
    """Set the terms the planner may schedule into. Called once from config; the module-level
    list is mutated in place so callers holding a reference see the change."""
    TERM_ORDER[:] = [t for t in ALL_TERMS if t in terms]


@dataclass(frozen=True)
class Course:
    code: str
    title: str
    credits: int
    prereqs: tuple[str, ...] = ()
    offered_terms: tuple[str, ...] = ()
    requirement_tags: tuple[str, ...] = ()
    workload_score: int = 3


@dataclass
class Profile:
    name: str = "Student"
    degree_program: str = "Computer Science"
    completed_courses: list[str] = field(default_factory=list)
    remaining_courses: list[str] = field(default_factory=list)
    start_term: str = "fall"
    start_year: int = 2026
    semesters_to_plan: int = 8
    max_credits_per_semester: int = 16

    def replace(self, **changes) -> "Profile":
        return Profile(**{**self.__dict__, **changes})


@dataclass
class Semester:
    term: str
    year: int
    courses: list[str]
    total_credits: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class Plan:
    semesters: list[Semester]
    unplanned_courses: list[str]
    warnings: list[str]


@dataclass
class Proposal:
    """Harness-side mirror of ``app.models.schemas.PlanEditProposal``."""

    rationale: str = ""
    reorder: list[str] = field(default_factory=list)
    defer: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    max_credits_per_semester: int | None = None


def next_term(term: str, year: int) -> tuple[str, int]:
    """Advance one term. A term that is not schedulable (summer, by default) is treated as
    'before the first schedulable term', so a summer start rolls forward to that fall rather
    than raising."""
    idx = TERM_ORDER.index(term) if term in TERM_ORDER else -1
    nxt = TERM_ORDER[(idx + 1) % len(TERM_ORDER)]
    return nxt, year + 1 if nxt == "spring" else year


def first_planning_term(term: str, year: int) -> tuple[str, int]:
    """Snap a start term onto the planning calendar.

    A student whose record says they start in summer still has to be given a plan; with summer
    excluded they simply start that fall. Silently scheduling INTO the summer term would be the
    bug, so the snap happens once, here, rather than being left to each caller."""
    return (term, year) if term in TERM_ORDER else next_term(term, year)


def prereqs_satisfied(course: Course, completed: set[str]) -> bool:
    return all(prereq in completed for prereq in course.prereqs)


def generate_plan(profile: Profile, catalog: list[Course]) -> Plan:
    """Port of ``app.services.planner.generate_plan``. Keep the two byte-for-byte equivalent
    in behaviour; the ordering key in particular is load-bearing for Mode A scores."""
    catalog_by_code = {course.code: course for course in catalog}
    completed = set(profile.completed_courses)
    remaining = list(profile.remaining_courses)
    warnings: list[str] = []
    semesters: list[Semester] = []

    unknown = [code for code in remaining if code not in catalog_by_code]
    if unknown:
        warnings.append(f"Unknown courses ignored: {', '.join(unknown)}")
        remaining = [code for code in remaining if code in catalog_by_code]

    priority = {code: index for index, code in enumerate(remaining)}

    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)

    for _ in range(profile.semesters_to_plan):
        selected: list[Course] = []
        selected_credits = 0
        semester_warnings: list[str] = []

        candidates = [
            catalog_by_code[code]
            for code in remaining
            if term in catalog_by_code[code].offered_terms
            and prereqs_satisfied(catalog_by_code[code], completed)
        ]
        candidates.sort(
            key=lambda c: (
                priority.get(c.code, len(priority)),
                "required" not in c.requirement_tags,
                c.workload_score,
                c.code,
            )
        )

        for course in candidates:
            if selected_credits + course.credits <= profile.max_credits_per_semester:
                selected.append(course)
                selected_credits += course.credits

        if not selected and remaining:
            blocked = []
            for code in remaining:
                course = catalog_by_code[code]
                reasons = []
                missing = [p for p in course.prereqs if p not in completed]
                if missing:
                    reasons.append(f"missing prereqs {list(missing)}")
                if term not in course.offered_terms:
                    reasons.append(f"not offered in {term}")
                if reasons:
                    blocked.append(f"{code}: {', '.join(reasons)}")
            if blocked:
                semester_warnings.append(
                    "No courses selected this term. Blocked courses: " + "; ".join(blocked)
                )

        planned = {course.code for course in selected}
        completed.update(planned)
        remaining = [code for code in remaining if code not in planned]

        semesters.append(
            Semester(
                term=term,
                year=year,
                courses=[course.code for course in selected],
                total_credits=selected_credits,
                warnings=semester_warnings,
            )
        )

        if not remaining:
            break
        term, year = next_term(term, year)

    if remaining:
        warnings.append(
            "Some courses could not be planned within the requested number of semesters: "
            + ", ".join(remaining)
        )

    return Plan(semesters=semesters, unplanned_courses=remaining, warnings=warnings)


def apply_proposal(profile: Profile, proposal: Proposal, catalog: list[Course]) -> Profile:
    """Port of ``advisor_agent._apply_proposal``. Everything is a preference, not a command:
    unknown codes drop out, reorder beats defer, and the cap falls back to the profile's."""
    by_code = {course.code: course for course in catalog}
    remaining = list(profile.remaining_courses)
    present = set(remaining)
    avoid = {tag.strip().lower() for tag in proposal.avoid_tags if tag.strip()}

    def is_avoided(code: str) -> bool:
        course = by_code.get(code)
        if course is None or not avoid:
            return False
        return any(tag.lower() in avoid for tag in course.requirement_tags)

    base = [c for c in remaining if not is_avoided(c)] + [c for c in remaining if is_avoided(c)]

    reorder = [c for c in proposal.reorder if c in present]
    reorder_set = set(reorder)
    defer = [c for c in proposal.defer if c in present and c not in reorder_set]
    pinned = reorder_set | set(defer)
    middle = [c for c in base if c not in pinned]

    cap = proposal.max_credits_per_semester or profile.max_credits_per_semester
    return profile.replace(
        remaining_courses=reorder + middle + defer,
        max_credits_per_semester=cap,
    )


def severity(plan: Plan) -> int:
    """Port of ``advisor_agent._severity`` — how incomplete a plan is."""
    return len(plan.unplanned_courses)
