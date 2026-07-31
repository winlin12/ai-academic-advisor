"""Deterministic direct-edit engine for generated plans (TODO Priority 2).

``POST /v1/plan/edit`` lands here. A student grabs one course and moves/adds/removes it;
this module applies that single mutation to the plan layout and re-validates the result —
prerequisites, term offerings, per-semester credit caps — completely offline from the LLM.

This is an *execution bridge* to :mod:`app.services.planner`, not a rewrite of it:

* ``planner.generate_plan`` stays the authority for *composing* a schedule from scratch
  (greedy reflow). It is deliberately **not** called here, because reflowing would undo the
  student's manual placement — the whole point of a direct edit is that the human chose the
  slot.
* ``planner.prereqs_satisfied`` remains the single prerequisite gate; this module reuses it
  verbatim so the legality rules can never drift between "generated" and "hand-edited"
  plans.

Validation is therefore *placement-preserving*: courses stay exactly where the student put
them, and rule violations surface as per-semester warnings rather than being silently
"fixed". Course facts (credits, prereqs, offered terms) come from the academic DB via
:func:`app.services.planner_catalog.fetch_courses_by_codes`, with the bundled fixture
catalog as the offline fallback — the same ladder the planner itself uses.
"""

from __future__ import annotations

import logging

import psycopg

from app.models.schemas import (
    Course,
    PlannedCourse,
    PlanResponse,
    SemesterPlan,
    StudentProfile,
)
from app.services.catalog import load_catalog
from app.services.planner import is_major_course, prereqs_satisfied
from app.services.planner_catalog import fetch_courses_by_codes, normalize_course_code

logger = logging.getLogger(__name__)


class PlanEditError(Exception):
    """Base class for rejected edits; messages are safe to surface to the client."""


class UnknownCourseError(PlanEditError):
    """The course code exists in neither the academic DB nor the fixture catalog."""


class CourseNotInPlanError(PlanEditError):
    """A move/remove named a course that is not scheduled anywhere in the plan."""


class DuplicateCourseError(PlanEditError):
    """An add named a course that is already scheduled or already completed."""


class InvalidTargetSemesterError(PlanEditError):
    """``target_semester`` does not index an existing semester in the plan."""


def resolve_edit_catalog(codes: set[str]) -> dict[str, Course]:
    """Build the validation catalog for an edit, keyed by *normalized* course code.

    Academic DB first (canonical codes, real credit hours), bundled fixture as fallback —
    both for a fully down DB and for individual codes the DB doesn't know (e.g. the demo
    profile's seed courses). Mirrors ``planner_catalog.resolve_profile_and_catalog``'s
    degradation ladder so edited plans and generated plans see the same course facts.
    """
    catalog: dict[str, Course] = {}
    try:
        catalog.update(fetch_courses_by_codes(codes))
    except psycopg.Error as exc:
        logger.warning("academic DB unavailable for plan editing; using bundled fixture: %s", exc)
    for course in load_catalog():
        catalog.setdefault(normalize_course_code(course.code), course)
    return catalog


def apply_plan_edit(
    plan: PlanResponse,
    operation: str,
    course_code: str,
    target_semester: int | None,
    profile: StudentProfile | None,
    catalog: dict[str, Course],
) -> PlanResponse:
    """Apply one move/add/remove to ``plan`` and return the re-validated result.

    Pure with respect to its inputs (``plan`` is never mutated) and does no I/O — the
    ``catalog`` is resolved by the caller so tests can inject the fixture directly.
    """
    semesters: list[list[PlannedCourse]] = [list(s.courses) for s in plan.semesters]
    unplanned = list(plan.unplanned_courses)
    wanted = normalize_course_code(course_code)

    def locate() -> tuple[int, int] | None:
        for sem_index, courses in enumerate(semesters):
            for course_index, planned in enumerate(courses):
                if normalize_course_code(planned.code) == wanted:
                    return sem_index, course_index
        return None

    def checked_target() -> int:
        if target_semester is None or not 0 <= target_semester < len(semesters):
            raise InvalidTargetSemesterError(
                f"target_semester must be between 0 and {len(semesters) - 1}"
            )
        return target_semester

    if operation == "remove":
        found = locate()
        if found is None:
            raise CourseNotInPlanError(f"'{course_code}' is not scheduled in this plan")
        sem_index, course_index = found
        removed = semesters[sem_index].pop(course_index)
        if all(normalize_course_code(code) != wanted for code in unplanned):
            unplanned.append(removed.code)

    elif operation == "move":
        target = checked_target()
        found = locate()
        if found is None:
            raise CourseNotInPlanError(f"'{course_code}' is not scheduled in this plan")
        sem_index, course_index = found
        semesters[target].append(semesters[sem_index].pop(course_index))

    elif operation == "add":
        target = checked_target()
        if locate() is not None:
            raise DuplicateCourseError(f"'{course_code}' is already scheduled in this plan")
        if profile is not None and wanted in {
            normalize_course_code(code) for code in profile.completed_courses
        }:
            raise DuplicateCourseError(f"'{course_code}' is already completed")
        course = catalog.get(wanted)
        if course is None:
            raise UnknownCourseError(f"'{course_code}' was not found in the course catalog")
        semesters[target].append(
            PlannedCourse(
                code=course.code,
                title=course.title,
                credits=course.credits,
                workload_score=course.workload_score,
            )
        )
        unplanned = [code for code in unplanned if normalize_course_code(code) != wanted]

    else:  # unreachable through the API (schema Literal), kept for direct callers
        raise PlanEditError(f"Unsupported operation '{operation}'")

    return _revalidate_layout(plan, semesters, unplanned, profile, catalog)


def _revalidate_layout(
    plan: PlanResponse,
    semesters: list[list[PlannedCourse]],
    unplanned: list[str],
    profile: StudentProfile | None,
    catalog: dict[str, Course],
) -> PlanResponse:
    """Re-derive totals and warnings for an edited layout without moving any course.

    Walks semesters chronologically, treating profile-completed courses plus everything
    scheduled in *earlier* semesters as satisfied prerequisites (same-semester courses do
    not count — you can't take a course and its prerequisite concurrently). Prerequisite
    legality is delegated to ``planner.prereqs_satisfied`` so the rule lives in one place.
    """
    completed: set[str] = (
        {normalize_course_code(code) for code in profile.completed_courses} if profile else set()
    )
    credit_cap = profile.max_credits_per_semester if profile else None

    rebuilt: list[SemesterPlan] = []
    for original, planned_courses in zip(plan.semesters, semesters):
        warnings: list[str] = []
        courses: list[PlannedCourse] = []
        total_credits = 0

        for planned in planned_courses:
            course = catalog.get(normalize_course_code(planned.code))
            if course is not None:
                # Refresh credits/title from the catalog so stale client copies self-heal.
                planned = PlannedCourse(
                    code=planned.code,
                    title=course.title,
                    credits=course.credits,
                    workload_score=course.workload_score,
                )
                normalized = course.model_copy(
                    update={"prereqs": [normalize_course_code(p) for p in course.prereqs]}
                )
                if not prereqs_satisfied(normalized, completed):
                    missing = [p for p in normalized.prereqs if p not in completed]
                    warnings.append(
                        f"{planned.code}: missing prerequisites ({', '.join(missing)})"
                    )
                if course.offered_terms and original.term not in course.offered_terms:
                    warnings.append(f"{planned.code}: not offered in {original.term}")
            else:
                warnings.append(
                    f"{planned.code}: not in catalog; keeping saved credits unverified"
                )
            courses.append(planned)
            total_credits += planned.credits

        if credit_cap is not None and total_credits > credit_cap:
            warnings.append(
                f"{total_credits} credits exceeds the {credit_cap}-credit cap for this semester"
            )

        # The second load cap, warned about on the same terms as the credit cap. A student who
        # drags a fourth CS course into a term is inside their credit cap and has still built a
        # semester the planner would never produce; without this the edit looks clean.
        if profile is not None:
            major_courses = sum(
                1 for planned in courses
                if is_major_course(planned.code, profile.major_subject)
            )
            if major_courses > profile.max_major_courses_per_semester:
                warnings.append(
                    f"{major_courses} {profile.major_subject} courses exceeds the "
                    f"{profile.max_major_courses_per_semester}-course limit for this semester"
                )
            elif major_courses > profile.preferred_major_courses_per_semester:
                warnings.append(
                    f"{major_courses} {profile.major_subject} courses in one semester — over "
                    f"the {profile.preferred_major_courses_per_semester} most students find "
                    f"manageable, though within the limit"
                )

        completed.update(normalize_course_code(course.code) for course in courses)
        rebuilt.append(
            SemesterPlan(
                term=original.term,
                year=original.year,
                courses=courses,
                total_credits=total_credits,
                average_workload=(
                    round(sum(course.workload_score for course in courses) / len(courses), 2)
                    if courses
                    else 0.0
                ),
                warnings=warnings,
            )
        )

    plan_warnings: list[str] = []
    if unplanned:
        plan_warnings.append(
            "Unscheduled courses (removed or not yet placed): " + ", ".join(unplanned)
        )

    return PlanResponse(
        student_name=plan.student_name,
        degree_program=plan.degree_program,
        semesters=rebuilt,
        unplanned_courses=unplanned,
        warnings=plan_warnings,
    )
