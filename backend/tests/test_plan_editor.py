"""Tests for the deterministic direct-edit engine (services/plan_editor.py).

Everything runs against the bundled fixture catalog injected directly into
``apply_plan_edit`` — no database, no llama.cpp, no HTTP. Baselines are produced by the real
``generate_plan`` so edits are always exercised against a plan the planner itself considers
legal.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import PlanEditRequest, StudentProfile
from app.services.catalog import load_catalog
from app.services.plan_editor import (
    CourseNotInPlanError,
    DuplicateCourseError,
    InvalidTargetSemesterError,
    UnknownCourseError,
    apply_plan_edit,
)
from app.services.planner import generate_plan
from app.services.planner_catalog import normalize_course_code


def fixture_catalog():
    return {normalize_course_code(course.code): course for course in load_catalog()}


def make_profile(**overrides) -> StudentProfile:
    defaults = dict(
        completed_courses=["CS251"],
        remaining_courses=["CS381", "CS502", "CS536", "CS541", "CS551", "CS577", "CS587"],
        start_term="fall",
        start_year=2026,
        semesters_to_plan=5,
        max_credits_per_semester=6,
    )
    defaults.update(overrides)
    return StudentProfile(**defaults)


def make_plan(profile: StudentProfile):
    return generate_plan(profile, load_catalog())


def find_semester(plan, code: str) -> int:
    for index, semester in enumerate(plan.semesters):
        if any(course.code == code for course in semester.courses):
            return index
    raise AssertionError(f"{code} not in plan")


def test_move_recalculates_credits_and_flags_violations():
    profile = make_profile()
    plan = make_plan(profile)
    source = find_semester(plan, "CS536")  # fall semester; CS536 is fall-only
    spring = next(
        i for i, s in enumerate(plan.semesters) if s.term == "spring" and s.courses
    )

    edited = apply_plan_edit(plan, "move", "CS536", spring, profile, fixture_catalog())

    # The course landed exactly where the student put it — no greedy reflow.
    assert any(c.code == "CS536" for c in edited.semesters[spring].courses)
    assert all(c.code != "CS536" for c in edited.semesters[source].courses)
    # Credit totals recalculated on both sides.
    assert edited.semesters[source].total_credits == plan.semesters[source].total_credits - 3
    assert edited.semesters[spring].total_credits == plan.semesters[spring].total_credits + 3
    # Violations surface as warnings instead of being silently "fixed".
    spring_warnings = " ".join(edited.semesters[spring].warnings)
    assert "CS536: not offered in spring" in spring_warnings
    assert f"exceeds the {profile.max_credits_per_semester}-credit cap" in spring_warnings


def test_move_before_prerequisite_warns():
    profile = make_profile()
    plan = make_plan(profile)
    cs577_at = find_semester(plan, "CS577")
    first = 0  # CS381 (CS577's prereq) is scheduled here — same semester must not count

    edited = apply_plan_edit(plan, "move", "CS577", first, profile, fixture_catalog())

    assert cs577_at != first
    assert any(
        w.startswith("CS577: missing prerequisites") for w in edited.semesters[first].warnings
    )


def test_move_normalizes_course_code_spelling():
    profile = make_profile()
    plan = make_plan(profile)
    target = len(plan.semesters) - 1

    edited = apply_plan_edit(plan, "move", "cs 536", target, profile, fixture_catalog())

    assert any(c.code == "CS536" for c in edited.semesters[target].courses)


def test_move_to_invalid_semester_raises():
    profile = make_profile()
    plan = make_plan(profile)

    with pytest.raises(InvalidTargetSemesterError):
        apply_plan_edit(plan, "move", "CS536", len(plan.semesters), profile, fixture_catalog())


def test_move_course_not_in_plan_raises():
    profile = make_profile()
    plan = make_plan(profile)

    with pytest.raises(CourseNotInPlanError):
        apply_plan_edit(plan, "move", "CS999", 0, profile, fixture_catalog())


def test_remove_sends_course_to_unplanned():
    profile = make_profile()
    plan = make_plan(profile)
    source = find_semester(plan, "CS502")

    edited = apply_plan_edit(plan, "remove", "CS502", None, profile, fixture_catalog())

    assert all(c.code != "CS502" for s in edited.semesters for c in s.courses)
    assert "CS502" in edited.unplanned_courses
    assert edited.semesters[source].total_credits == plan.semesters[source].total_credits - 3
    assert any("Unscheduled courses" in w for w in edited.warnings)


def test_add_restores_removed_course_and_clears_unplanned():
    profile = make_profile()
    plan = make_plan(profile)
    source = find_semester(plan, "CS502")

    removed = apply_plan_edit(plan, "remove", "CS502", None, profile, fixture_catalog())
    # Lowercase/spacing differences must still match the catalog entry.
    restored = apply_plan_edit(removed, "add", "cs502", source, profile, fixture_catalog())

    assert any(c.code == "CS502" for c in restored.semesters[source].courses)
    assert "CS502" not in restored.unplanned_courses
    added = next(c for c in restored.semesters[source].courses if c.code == "CS502")
    assert added.title == "Compilers"  # facts come from the catalog, not the client
    assert added.credits == 3


def test_add_duplicate_and_completed_and_unknown_raise():
    profile = make_profile()
    plan = make_plan(profile)

    with pytest.raises(DuplicateCourseError):
        apply_plan_edit(plan, "add", "CS536", 0, profile, fixture_catalog())
    with pytest.raises(DuplicateCourseError):
        apply_plan_edit(plan, "add", "CS251", 0, profile, fixture_catalog())
    with pytest.raises(UnknownCourseError):
        apply_plan_edit(plan, "add", "CS999", 0, profile, fixture_catalog())


def test_edit_does_not_mutate_input_plan():
    profile = make_profile()
    plan = make_plan(profile)
    snapshot = plan.model_dump()

    apply_plan_edit(plan, "remove", "CS502", None, profile, fixture_catalog())

    assert plan.model_dump() == snapshot


def test_edit_without_profile_still_validates_structure():
    plan = make_plan(make_profile())
    spring = next(
        i for i, s in enumerate(plan.semesters) if s.term == "spring" and s.courses
    )

    edited = apply_plan_edit(plan, "move", "CS536", spring, None, fixture_catalog())

    # Term-offering warnings don't need a profile; credit-cap warnings do.
    warnings = " ".join(edited.semesters[spring].warnings)
    assert "CS536: not offered in spring" in warnings
    assert "credit cap" not in warnings


def test_request_schema_requires_target_for_move_and_add():
    plan = make_plan(make_profile())

    with pytest.raises(ValidationError):
        PlanEditRequest(plan=plan, operation="move", course_code="CS536")
    with pytest.raises(ValidationError):
        PlanEditRequest(plan=plan, operation="add", course_code="CS502")
    # remove needs no target
    PlanEditRequest(plan=plan, operation="remove", course_code="CS536")
