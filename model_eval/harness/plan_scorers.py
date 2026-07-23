"""Plan-of-study scoring. The headline instrument of this harness.

Unlike faithfulness (a heuristic that needs your eyes), everything here is FULLY AUTOMATIC
and decidable: a plan either schedules CS 38100 before its prerequisites or it doesn't. The
only judgment call baked in is the fixture itself — see the provenance header in
``plan_fixtures/cs_machine_intelligence.yaml``. Wrong fixture rows bias every model the same
direction, which keeps rankings usable and absolute numbers suspect.

WHAT IS MEASURED

  Hard violations (any one of these makes a plan NOT viable — a student following it would
  hit a registration wall or fail to graduate):

    hallucinated_course      a code that does not exist in the catalog fixture
    prereq_violation         scheduled in a term where a prerequisite is not yet complete;
                             same-semester prerequisites do NOT count as satisfied
    term_offering_violation  scheduled in a term the course is not offered
    credit_cap_violation     semester credits exceed the profile's cap
    duplicate_course         the same course twice, or a course already completed

  Completeness:

    requirement_coverage     fraction of the fixture's requirement groups satisfied by
                             completed + planned courses
    missing_requirements     which groups are short, and by what

  PLAN_VIABLE = zero hard violations AND requirement_coverage == 1.0 within the horizon.
  That conjunction is the number to compare models on. ``violations`` is reported alongside
  it so a near-miss (one bad term offering) is visibly different from a plan that invented
  six courses.

DELIBERATELY NOT SCORED HERE: whether the plan is *pleasant* — workload balance, spreading
theory courses, summer usage. Those are preferences, not correctness, and rolling them into
the headline number would let a model trade a real prerequisite violation against a nicer
credit distribution. They are recorded as diagnostics (``credit_spread``) and left out of
PLAN_VIABLE on purpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .fixtures import Fixture
from .planner import Plan, Profile, first_planning_term, next_term

_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?-?\s?(\d{3,5})\b")


def normalize_code(code: str) -> str:
    """'cs25100', 'CS-25100', 'CS 251' -> 'CS 25100'-shaped. Purely lexical.

    Models are not being graded on spacing. A three-digit legacy code is left as-is and will
    simply fail the catalog lookup — that IS the right outcome, since 'CS 251' is not a code
    the current catalog can register you for.
    """
    compact = "".join(str(code).split()).upper().replace("-", "")
    for index, char in enumerate(compact):
        if char.isdigit():
            return compact if index == 0 else f"{compact[:index]} {compact[index:]}"
    return compact


@dataclass
class PlanScore:
    structure_ok: bool = False          # did we get a parseable plan at all?
    viable: bool = False
    violations: list[str] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = field(default_factory=list)
    planned_credits: int = 0
    semester_credits: list[int] = field(default_factory=list)
    credit_spread: int = 0              # max - min across non-empty semesters (diagnostic)
    hallucinated: list[str] = field(default_factory=list)
    assertions_passed: dict[str, bool] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
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
            "hallucinated_courses": self.hallucinated,
            "assertions_passed": self.assertions_passed,
        }


def _terms(profile: Profile, count: int) -> list[tuple[str, int]]:
    """The calendar a plan's Nth semester actually falls in.

    Must snap the start term the same way the planner does. When these two disagree the
    scorer grades semester N against the wrong term and reports term-offering violations for
    a schedule that is perfectly legal — which is exactly what happened the first time summer
    was switched off.
    """
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(count):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def requirement_coverage(
    fixture: Fixture, satisfied: set[str]
) -> tuple[float, list[str]]:
    """Fraction of requirement groups met, plus a human-readable list of what's short."""
    groups = fixture.requirement_groups
    if not groups:
        return 1.0, []
    met = 0
    missing: list[str] = []
    for group in groups:
        courses = group.get("courses", [])
        if group.get("kind") == "all":
            short = [c for c in courses if c not in satisfied]
            if not short:
                met += 1
            else:
                missing.append(f"{group['id']}: missing {', '.join(short)}")
        else:
            have = sum(fixture.credits(c) for c in courses if c in satisfied)
            target = float(group.get("choose_credits") or 0)
            if have >= target:
                met += 1
            else:
                missing.append(
                    f"{group['id']}: {have:g}/{target:g} credits "
                    f"(options: {', '.join(courses)})"
                )
    return met / len(groups), missing


def score_plan(
    fixture: Fixture,
    profile: Profile,
    semesters: list[list[str]],
    *,
    assertions: list[dict[str, Any]] | None = None,
) -> PlanScore:
    """Score a schedule expressed as a list of per-semester course-code lists.

    Works for BOTH modes: Mode A passes the deterministic planner's output (so hard
    violations should be zero by construction — if they aren't, the harness's planner port
    has drifted from the app's and that is worth knowing), Mode B passes whatever JSON the
    model produced.
    """
    score = PlanScore(structure_ok=True)
    by_code = fixture.by_code
    counts: dict[str, int] = {}

    def flag(kind: str, detail: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        score.violations.append(f"{kind}: {detail}")

    completed = {normalize_code(c) for c in profile.completed_courses}
    seen: set[str] = set()
    terms = _terms(profile, len(semesters))

    for index, (raw_codes, (term, year)) in enumerate(zip(semesters, terms)):
        codes = [normalize_code(c) for c in raw_codes]
        credits = 0
        # Prereqs are evaluated against courses completed BEFORE this semester: a course
        # taken alongside its own prerequisite is a scheduling error, not a co-requisite.
        available = set(completed)

        for code in codes:
            course = by_code.get(code)
            if course is None:
                score.hallucinated.append(code)
                flag("hallucinated_course", f"{code} (semester {index + 1}) is not in the catalog")
                continue
            if code in seen:
                flag("duplicate_course", f"{code} appears more than once")
            elif code in completed:
                flag("duplicate_course", f"{code} was already completed before the plan starts")
            seen.add(code)

            credits += course.credits
            if course.offered_terms and term not in course.offered_terms:
                flag(
                    "term_offering_violation",
                    f"{code} in {term} {year}; offered {'/'.join(course.offered_terms)}",
                )
            missing = [p for p in course.prereqs if normalize_code(p) not in available]
            if missing:
                flag(
                    "prereq_violation",
                    f"{code} in semester {index + 1} needs {', '.join(missing)} first",
                )

        score.semester_credits.append(credits)
        if credits > profile.max_credits_per_semester:
            flag(
                "credit_cap_violation",
                f"semester {index + 1} has {credits} credits, cap is "
                f"{profile.max_credits_per_semester}",
            )
        completed |= {c for c in codes if c in by_code}

    score.planned_credits = sum(score.semester_credits)
    nonempty = [c for c in score.semester_credits if c > 0]
    score.credit_spread = (max(nonempty) - min(nonempty)) if nonempty else 0

    satisfied = {normalize_code(c) for c in profile.completed_courses} | seen
    score.requirement_coverage, score.missing_requirements = requirement_coverage(
        fixture, satisfied
    )
    score.violation_counts = counts
    score.viable = not counts and score.requirement_coverage >= 1.0

    if assertions:
        score.assertions_passed = check_assertions(assertions, semesters, score)
    return score


# --- scenario assertions -------------------------------------------------------------------
# Mode A's problem: the deterministic planner guarantees legality, so plan viability alone
# cannot separate a model that understood the student from one that returned an empty
# proposal. These assertions ask the only question left — was the student's stated ask
# actually acted on?


def check_assertions(
    assertions: list[dict[str, Any]],
    semesters: list[list[str]],
    score: PlanScore,
    proposal: dict[str, Any] | None = None,
) -> dict[str, bool]:
    placement: dict[str, int] = {}
    for index, codes in enumerate(semesters):
        for code in codes:
            placement.setdefault(normalize_code(code), index)

    results: dict[str, bool] = {}
    for spec in assertions:
        kind = spec.get("kind")
        if kind == "scheduled_before":
            code = normalize_code(spec["course"])
            index = placement.get(code)
            key = f"{kind}:{code}<{spec['semester_index']}"
            results[key] = index is not None and index < int(spec["semester_index"])
        elif kind == "any_scheduled_before":
            codes = [normalize_code(c) for c in spec["courses"]]
            key = f"{kind}:{'|'.join(codes)}<{spec['semester_index']}"
            results[key] = any(
                placement.get(c) is not None and placement[c] < int(spec["semester_index"])
                for c in codes
            )
        elif kind == "max_credits_at_most":
            key = f"{kind}:{spec['value']}"
            results[key] = all(c <= int(spec["value"]) for c in score.semester_credits)
        elif kind == "proposal_sets_credit_cap":
            # Mode B has no proposal; the assertion is not applicable and is skipped rather
            # than scored as a failure (counting N/A as a miss would penalise Mode B for a
            # question it was never asked).
            if proposal is not None:
                key = f"{kind}:{spec['value']}"
                results[key] = proposal.get("max_credits_per_semester") == int(spec["value"])
    return results


# --- Mode A: proposal quality ---------------------------------------------------------------


def score_proposal(
    proposal: dict[str, Any], profile: Profile, fixture: Fixture
) -> dict[str, Any]:
    """Is the model's PlanEditProposal GROUNDED — does it name things that exist?

    The app degrades a hallucinated proposal to a no-op by design, so a bad proposal never
    produces an illegal plan; it just produces a plan that ignored the student. This is where
    that silent failure becomes visible.
    """
    remaining = {normalize_code(c) for c in profile.remaining_courses}
    known_tags = {tag.lower() for course in fixture.catalog for tag in course.requirement_tags}

    reorder = [normalize_code(c) for c in proposal.get("reorder") or []]
    defer = [normalize_code(c) for c in proposal.get("defer") or []]
    tags = [str(t).strip().lower() for t in proposal.get("avoid_tags") or []]

    # Truncated: the most common failure here is a model writing a whole semester layout
    # ("fall 2026: CS 25000, CS 25100, ... [14 cr]") into a field that expects one course
    # code. Those strings are enormous and would otherwise dominate the JSONL, so keep enough
    # to recognise the shape and no more.
    ungrounded_codes = [c[:60] for c in reorder + defer if c not in remaining]
    ungrounded_tags = [t[:60] for t in tags if t and t not in known_tags]
    conflicts = sorted(set(reorder) & set(defer))
    cap = proposal.get("max_credits_per_semester")

    touched = bool(reorder or defer or tags or cap is not None)
    return {
        "proposal_touched_anything": touched,
        "proposal_grounded": not ungrounded_codes and not ungrounded_tags and not conflicts,
        "ungrounded_codes": ungrounded_codes,
        "ungrounded_tags": ungrounded_tags,
        "reorder_defer_conflict": conflicts,
        "proposed_credit_cap": cap,
        "rationale_len": len(str(proposal.get("rationale") or "")),
    }


def rationale_flags(rationale: str, planned_codes: set[str], fixture: Fixture) -> list[str]:
    """Course codes the rationale claims about that aren't in the plan or the catalog.

    Same triage-only status as ``scorers.faithfulness_flags``: it catches entity invention,
    not "CS 38100 will be easier in the spring"-style relational fiction.
    """
    flags: list[str] = []
    for subject, number in _COURSE_CODE_RE.findall(rationale.upper()):
        code = normalize_code(f"{subject} {number}")
        if code not in fixture.by_code:
            flags.append(f"rationale cites unknown course: {code}")
        elif code not in planned_codes:
            flags.append(f"rationale cites course not in the plan: {code}")
    return flags


def plan_to_semesters(plan: Plan) -> list[list[str]]:
    return [semester.courses for semester in plan.semesters]
